#pragma once

#include "vec.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <string_view>

#include "simdjson.h"

namespace rinha {

struct MccRisk { uint16_t mcc; float risk; };

inline constexpr std::array kMccTable = std::to_array<MccRisk>({
    {4511, 0.35f}, {5311, 0.25f}, {5411, 0.15f}, {5812, 0.30f},
    {5912, 0.20f}, {5944, 0.45f}, {5999, 0.50f}, {7801, 0.80f},
    {7802, 0.75f}, {7995, 0.85f},
});
inline constexpr float kMccDefault = 0.50f;

inline float mcc_risk(std::string_view mcc_str) noexcept {
    if (mcc_str.size() != 4) return kMccDefault;
    uint16_t mcc = 0;
    for (char c : mcc_str) {
        if (c < '0' || c > '9') return kMccDefault;
        mcc = static_cast<uint16_t>(mcc * 10 + (c - '0'));
    }
    auto it = std::ranges::lower_bound(kMccTable, mcc, {}, &MccRisk::mcc);
    if (it != kMccTable.end() && it->mcc == mcc) return it->risk;
    return kMccDefault;
}

inline constexpr int parse_hour(std::string_view ts) noexcept {
    if (ts.size() < 16) return 0;
    return (ts[11] - '0') * 10 + (ts[12] - '0');
}

inline constexpr int day_of_week(std::string_view ts) noexcept {
    if (ts.size() < 10) return 0;
    int y = (ts[0]-'0')*1000 + (ts[1]-'0')*100 + (ts[2]-'0')*10 + (ts[3]-'0');
    int m = (ts[5]-'0')*10 + (ts[6]-'0');
    int d = (ts[8]-'0')*10 + (ts[9]-'0');
    constexpr int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    if (m < 3) y -= 1;
    return ((y + y/4 - y/100 + y/400 + t[m-1] + d) % 7 + 6) % 7;
}

inline float minutes_between(std::string_view ts1, std::string_view ts2) noexcept {
    auto to_minutes = [](std::string_view t) -> int64_t {
        if (t.size() < 19) return 0;
        int Y = (t[0]-'0')*1000 + (t[1]-'0')*100 + (t[2]-'0')*10 + (t[3]-'0');
        int M = (t[5]-'0')*10 + (t[6]-'0');
        int D = (t[8]-'0')*10 + (t[9]-'0');
        int h = (t[11]-'0')*10 + (t[12]-'0');
        int m = (t[14]-'0')*10 + (t[15]-'0');
        constexpr int dm[] = {0,31,59,90,120,151,181,212,243,273,304,334};
        int64_t days = int64_t(Y)*365 + Y/4 - Y/100 + Y/400 + dm[M-1] + D;
        if (M <= 2 && Y%4 == 0 && (Y%100 != 0 || Y%400 == 0)) --days;
        return days * 1440 + h * 60 + m;
    };
    return static_cast<float>(std::abs(to_minutes(ts1) - to_minutes(ts2)));
}

class Vectorizer {
public:
    Vectorizer() = default;

    [[gnu::hot]]
    bool vectorize(const char* json, std::size_t len,
                   Vec16& vec_out, uint8_t& key_out) noexcept {
        alignas(32) std::array<float, kDimsPad> query{};
        if (!vectorize_f32(json, len, query)) return false;
        quantize_vec(query, vec_out);
        key_out = compute_partition_key(query);
        return true;
    }

    [[gnu::hot]]
    bool vectorize_f32(const char* json, std::size_t len,
                       std::array<float, kDimsPad>& query) noexcept {
        auto result = parser_.iterate(json, len, len + simdjson::SIMDJSON_PADDING);
        if (result.error()) return false;
        auto doc = std::move(result.value_unsafe());

        double amount{}, installments{}, avg_amount{}, tx_count_24h{};
        double merchant_avg{}, km_from_home{}, km_from_current{};
        std::string_view requested_at, merchant_id, mcc_str, last_ts;
        bool is_online{}, card_present{}, has_last{}, unknown_merchant{true};

        {
            auto tx = doc["transaction"];
            amount       = tx["amount"].get_double().value_unsafe();
            installments = tx["installments"].get_double().value_unsafe();
            requested_at = tx["requested_at"].get_string().value_unsafe();
        }

        {
            auto cust = doc["customer"];
            avg_amount   = cust["avg_amount"].get_double().value_unsafe();
            tx_count_24h = cust["tx_count_24h"].get_double().value_unsafe();

            known_count_ = 0;
            if (auto arr = cust["known_merchants"].get_array(); !arr.error()) {
                for (auto m : arr.value_unsafe()) {
                    if (auto sv = m.get_string(); !sv.error() && known_count_ < kMaxKnown)
                        known_[known_count_++] = sv.value_unsafe();
                }
            }
        }

        {
            auto merch = doc["merchant"];
            merchant_id  = merch["id"].get_string().value_unsafe();
            mcc_str      = merch["mcc"].get_string().value_unsafe();
            merchant_avg = merch["avg_amount"].get_double().value_unsafe();
        }

        unknown_merchant = std::none_of(
            known_.begin(), known_.begin() + known_count_,
            [&](std::string_view sv) { return sv == merchant_id; });

        {
            auto term = doc["terminal"];
            is_online    = term["is_online"].get_bool().value_unsafe();
            card_present = term["card_present"].get_bool().value_unsafe();
            km_from_home = term["km_from_home"].get_double().value_unsafe();
        }

        {
            auto last = doc["last_transaction"];
            if (auto lt = last.type(); !lt.error() && lt.value_unsafe() == simdjson::ondemand::json_type::object) {
                has_last = true;
                last_ts         = last["timestamp"].get_string().value_unsafe();
                km_from_current = last["km_from_current"].get_double().value_unsafe();
            }
        }

        float a = static_cast<float>(amount);
        float avg = static_cast<float>(avg_amount);

        query = {{
            clamp01(a / 10000.0f),
            clamp01(static_cast<float>(installments) / 12.0f),
            clamp01((avg > 0.0f ? a / avg : 0.0f) / 10.0f),
            static_cast<float>(parse_hour(requested_at)) / 23.0f,
            static_cast<float>(day_of_week(requested_at)) / 6.0f,
            has_last ? clamp01(minutes_between(requested_at, last_ts) / 1440.0f) : -1.0f,
            has_last ? clamp01(static_cast<float>(km_from_current) / 1000.0f) : -1.0f,
            clamp01(static_cast<float>(km_from_home) / 1000.0f),
            clamp01(static_cast<float>(tx_count_24h) / 20.0f),
            is_online ? 1.0f : 0.0f,
            card_present ? 1.0f : 0.0f,
            unknown_merchant ? 1.0f : 0.0f,
            mcc_risk(mcc_str),
            clamp01(static_cast<float>(merchant_avg) / 10000.0f),
            0.0f, 0.0f
        }};

        return true;
    }

private:
    static constexpr int kMaxKnown = 64;
    simdjson::ondemand::parser parser_;
    std::array<std::string_view, kMaxKnown> known_{};
    int known_count_ = 0;
};

} // namespace rinha
