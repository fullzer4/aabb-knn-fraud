#pragma once

#include "vec.hpp"

#include <cstring>
#include <string_view>

#include "simdjson.h"

namespace rinha {

struct MccRisk {
    uint16_t mcc;
    float    risk;
};

inline constexpr MccRisk kMccTable[] = {
    {4511, 0.35f}, {5311, 0.25f}, {5411, 0.15f}, {5812, 0.30f},
    {5912, 0.20f}, {5944, 0.45f}, {5999, 0.50f}, {7801, 0.80f},
    {7802, 0.75f}, {7995, 0.85f},
};
inline constexpr float kMccDefault = 0.50f;

inline float mcc_risk(std::string_view mcc_str) {
    if (mcc_str.size() != 4) return kMccDefault;
    uint16_t mcc = 0;
    for (char c : mcc_str) {
        if (c < '0' || c > '9') return kMccDefault;
        mcc = static_cast<uint16_t>(mcc * 10 + (c - '0'));
    }
    for (const auto& e : kMccTable)
        if (e.mcc == mcc) return e.risk;
    return kMccDefault;
}

inline int parse_hour(std::string_view ts) {
    if (ts.size() < 16) return 0;
    return (ts[11] - '0') * 10 + (ts[12] - '0');
}

// Tomohiko Sakamoto's algorithm, returns mon=0..sun=6
inline int day_of_week(std::string_view ts) {
    if (ts.size() < 10) return 0;
    int y = (ts[0] - '0') * 1000 + (ts[1] - '0') * 100 +
            (ts[2] - '0') * 10   + (ts[3] - '0');
    int m = (ts[5] - '0') * 10 + (ts[6] - '0');
    int d = (ts[8] - '0') * 10 + (ts[9] - '0');

    static constexpr int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    if (m < 3) y -= 1;
    int dow = (y + y/4 - y/100 + y/400 + t[m-1] + d) % 7;
    // Sakamoto: 0=Sun, 1=Mon, ..., 6=Sat → we want Mon=0, Sun=6
    return (dow + 6) % 7;
}

// Minutes between two ISO 8601 timestamps
inline float minutes_between(std::string_view ts1, std::string_view ts2) {
    // Quick parse: YYYY-MM-DDThh:mm:ssZ
    auto parse_epoch_minutes = [](std::string_view t) -> int64_t {
        if (t.size() < 19) return 0;
        int Y = (t[0]-'0')*1000 + (t[1]-'0')*100 + (t[2]-'0')*10 + (t[3]-'0');
        int M = (t[5]-'0')*10 + (t[6]-'0');
        int D = (t[8]-'0')*10 + (t[9]-'0');
        int h = (t[11]-'0')*10 + (t[12]-'0');
        int m = (t[14]-'0')*10 + (t[15]-'0');

        // Approximate: days since epoch (good enough for diff)
        static constexpr int days_before_month[] =
            {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
        int64_t days = static_cast<int64_t>(Y) * 365 + Y/4 - Y/100 + Y/400
                     + days_before_month[M-1] + D;
        if (M <= 2 && Y % 4 == 0 && (Y % 100 != 0 || Y % 400 == 0))
            days -= 1;
        return days * 1440 + h * 60 + m;
    };

    int64_t m1 = parse_epoch_minutes(ts1);
    int64_t m2 = parse_epoch_minutes(ts2);
    int64_t diff = m1 - m2;
    if (diff < 0) diff = -diff;
    return static_cast<float>(diff);
}

// ── Vectorize a JSON request body → 14-dim float32 ───────────────

class Vectorizer {
public:
    Vectorizer() = default;

    // Parse JSON body and fill query[0..kDimsPad-1] (padded)
    // Returns true on success
    [[gnu::hot]]
    bool vectorize(const char* json, std::size_t len, float* query) {
        auto result = parser_.iterate(json, len, len + simdjson::SIMDJSON_PADDING);
        if (result.error()) return false;
        auto doc = std::move(result.value_unsafe());

        // ── transaction ───────────────────────────────────
        double amount = 0, installments = 0;
        std::string_view requested_at;
        {
            auto tx = doc["transaction"];
            amount       = tx["amount"].get_double().value_unsafe();
            installments = tx["installments"].get_double().value_unsafe();
            requested_at = tx["requested_at"].get_string().value_unsafe();
        }

        // ── customer ──────────────────────────────────────
        double avg_amount = 0;
        double tx_count_24h = 0;
        {
            auto cust = doc["customer"];
            avg_amount   = cust["avg_amount"].get_double().value_unsafe();
            tx_count_24h = cust["tx_count_24h"].get_double().value_unsafe();
        }

        // ── merchant ──────────────────────────────────────
        std::string_view merchant_id, mcc_str;
        double merchant_avg = 0;
        {
            auto merch = doc["merchant"];
            merchant_id  = merch["id"].get_string().value_unsafe();
            mcc_str      = merch["mcc"].get_string().value_unsafe();
            merchant_avg = merch["avg_amount"].get_double().value_unsafe();
        }

        // ── terminal ──────────────────────────────────────
        bool is_online = false, card_present = false;
        double km_from_home = 0;
        {
            auto term = doc["terminal"];
            is_online    = term["is_online"].get_bool().value_unsafe();
            card_present = term["card_present"].get_bool().value_unsafe();
            km_from_home = term["km_from_home"].get_double().value_unsafe();
        }

        // ── last_transaction (nullable) ───────────────────
        bool has_last = false;
        std::string_view last_ts;
        double km_from_current = 0;
        {
            auto last = doc["last_transaction"];
            auto last_type = last.type();
            if (!last_type.error() && last_type.value_unsafe() == simdjson::ondemand::json_type::object) {
                has_last = true;
                last_ts         = last["timestamp"].get_string().value_unsafe();
                km_from_current = last["km_from_current"].get_double().value_unsafe();
            }
        }

        // ── unknown_merchant check ────────────────────────
        bool unknown_merchant = true;
        {
            // Re-iterate to access known_merchants (simdjson on-demand is forward-only)
            auto result2 = parser2_.iterate(json, len, len + simdjson::SIMDJSON_PADDING);
            if (!result2.error()) {
                auto doc2 = std::move(result2.value_unsafe());
                auto known = doc2["customer"]["known_merchants"];
                auto arr = known.get_array().value_unsafe();
                for (auto m : arr) {
                    auto sv = m.get_string().value_unsafe();
                    if (sv == merchant_id) {
                        unknown_merchant = false;
                        break;
                    }
                }
            }
        }

        // ── Build 14-dim vector ───────────────────────────
        float a = static_cast<float>(amount);
        float avg = static_cast<float>(avg_amount);

        query[0]  = clamp01(a / 10000.0f);                                       // amount
        query[1]  = clamp01(static_cast<float>(installments) / 12.0f);            // installments
        query[2]  = clamp01((avg > 0.0f ? a / avg : 0.0f) / 10.0f);              // amount_vs_avg
        query[3]  = static_cast<float>(parse_hour(requested_at)) / 23.0f;         // hour_of_day
        query[4]  = static_cast<float>(day_of_week(requested_at)) / 6.0f;         // day_of_week
        query[5]  = has_last ? clamp01(minutes_between(requested_at, last_ts) / 1440.0f) : -1.0f;
        query[6]  = has_last ? clamp01(static_cast<float>(km_from_current) / 1000.0f)    : -1.0f;
        query[7]  = clamp01(static_cast<float>(km_from_home) / 1000.0f);          // km_from_home
        query[8]  = clamp01(static_cast<float>(tx_count_24h) / 20.0f);            // tx_count_24h
        query[9]  = is_online ? 1.0f : 0.0f;                                     // is_online
        query[10] = card_present ? 1.0f : 0.0f;                                  // card_present
        query[11] = unknown_merchant ? 1.0f : 0.0f;                              // unknown_merchant
        query[12] = mcc_risk(mcc_str);                                           // mcc_risk
        query[13] = clamp01(static_cast<float>(merchant_avg) / 10000.0f);         // merchant_avg

        // Pad to 16 dims
        query[14] = 0.0f;
        query[15] = 0.0f;

        return true;
    }

private:
    simdjson::ondemand::parser parser_;
    simdjson::ondemand::parser parser2_;
};

} // namespace rinha
