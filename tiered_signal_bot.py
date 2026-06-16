"""
╔══════════════════════════════════════════════════════════════╗
║  Katmanlı Sinyal Botu — PEPE & PENGU                         ║
║  1dk + 5dk + 15dk + 1sa kombinasyonu                         ║
║  Amaç: Daha sık sinyal, ama PARA KAYBETMEMEYİ önceleyen      ║
║  3 katman: Premium / Standart / Hızlı Scalp                  ║
╚══════════════════════════════════════════════════════════════╝

Kurulum:
    pip install ccxt pandas pandas-ta python-telegram-bot

Çalıştırma:
    python tiered_signal_bot.py
"""

import asyncio
import logging
from datetime import datetime

import ccxt
import pandas as pd
import pandas_ta as ta
from telegram import Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
import os
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "BURAYA_TOKEN")
CHAT_ID        = os.environ.get("CHAT_ID", "BURAYA_CHAT_ID")

# Takip edilen coinler — her biri için ayrı eşik
COINS = {
    "PEPE/USDT": {
        "long_rsi_premium": 30, "long_rsi_standard": 38,
        "short_rsi_premium": 70, "short_rsi_standard": 62,
    },
    "PENGU/USDT": {
        "long_rsi_premium": 25, "long_rsi_standard": 38,
        "short_rsi_premium": 75, "short_rsi_standard": 62,
    },
}
BTC_SYMBOL = "BTC/USDT"

CHECK_INTERVAL = 30   # saniye — sık kontrol (1dk/5dk verisi olduğu için)

# ── Risk yönetimi — PARA KAYBETMEME kuralları ───────────────────
MAX_RISK_PER_TRADE_PCT = 1.5     # Sermayenin max %1.5'i risk
DAILY_MAX_LOSS_PCT     = 5.0     # Günlük max kayıp limiti
CONSECUTIVE_LOSS_PAUSE = 3       # 3 ardışık kayıpta dur

# ── İndikatör periyotları ────────────────────────────────────────
RSI_LEN, EMA_FAST, EMA_SLOW = 14, 9, 21
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
BB_LEN, BB_STD = 20, 2.0
ATR_LEN, VOL_MA_LEN = 14, 20
STOCH_K, STOCH_D = 14, 3
# ═══════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

exchange = ccxt.binance({"enableRateLimit": True})

# Durum takibi
last_signals: dict = {}          # {symbol: {"type":..., "price":..., "time":...}}
daily_pnl_tracker = {"loss_pct": 0.0, "consecutive_losses": 0, "date": datetime.now().date()}


# ─────────────────────────────────────────────────────────────
#  VERİ & İNDİKATÖR
# ─────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, tf: str, limit: int = 100) -> pd.DataFrame:
    bars = exchange.fetch_ohlcv(symbol, tf, limit=limit)
    df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["rsi"]       = ta.rsi(c, length=RSI_LEN)
    df["ema_fast"]  = ta.ema(c, length=EMA_FAST)
    df["ema_slow"]  = ta.ema(c, length=EMA_SLOW)
    macd            = ta.macd(c, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIG)
    df["macd_hist"] = macd.iloc[:, 1]
    bb              = ta.bbands(c, length=BB_LEN, std=BB_STD)
    df["bb_up"]     = bb.iloc[:, 2]
    df["bb_low"]    = bb.iloc[:, 0]
    df["bb_mid"]    = bb.iloc[:, 1]
    df["atr"]       = ta.atr(h, l, c, length=ATR_LEN)
    df["vol_ma"]    = v.rolling(VOL_MA_LEN).mean()
    df["vol_ratio"] = v / df["vol_ma"]
    stoch           = ta.stoch(h, l, c, k=STOCH_K, d=STOCH_D)
    df["stoch_k"]   = stoch.iloc[:, 0]
    df["stoch_d"]   = stoch.iloc[:, 1]
    df["is_green"]  = c > df["open"]
    return df


# ─────────────────────────────────────────────────────────────
#  3 KATMANLI SİNYAL MOTORU
# ─────────────────────────────────────────────────────────────

def evaluate_tiered(symbol: str, dfs: dict, btc_15m: pd.DataFrame) -> dict | None:
    """
    dfs = {"1m":..., "5m":..., "15m":..., "1h":...}
    Üç katmanı sırayla kontrol eder, en güçlüsünü döndürür.
    """
    cfg = COINS[symbol]
    m1, m5, m15, h1 = dfs["1m"].iloc[-1], dfs["5m"].iloc[-1], dfs["15m"].iloc[-1], dfs["1h"].iloc[-1]
    m1p, m5p = dfs["1m"].iloc[-2], dfs["5m"].iloc[-2]
    btc = btc_15m.iloc[-1]
    price = m1["close"]

    if any(pd.isna(x) for x in [h1["rsi"], m15["rsi"], m5["rsi"], btc["rsi"]]):
        return None

    # ════════ KATMAN 1: PREMIUM (1H + 15M + BTC) ════════════════
    long_premium = {
        "1H RSI ekstrem":     h1["rsi"] < cfg["long_rsi_premium"],
        "1H BB alt bant":     price <= h1["bb_low"] * 1.008,
        "Hacim 2.5x+":        h1["vol_ratio"] >= 2.5,
        "15M MACD/EMA teyit": (m15["macd_hist"] > 0) or (m15["ema_fast"] > m15["ema_slow"]),
        "BTC engel yok":      btc["rsi"] > 35,
    }
    short_premium = {
        "1H RSI ekstrem":     h1["rsi"] > cfg["short_rsi_premium"],
        "1H BB üst bant":     price >= h1["bb_up"] * 0.992,
        "Hacim 2.5x+":        h1["vol_ratio"] >= 2.5,
        "15M MACD/EMA teyit": (m15["macd_hist"] < 0) or (m15["ema_fast"] < m15["ema_slow"]),
        "BTC de baskıda":     btc["rsi"] > 55 or btc["ema_fast"] < btc["ema_slow"],
    }
    lp_score, sp_score = sum(long_premium.values()), sum(short_premium.values())

    if lp_score >= 4:
        return build_signal(symbol, "LONG", "PREMIUM", price, h1, m1,
                             {k: v for k, v in long_premium.items() if v}, lp_score, 5)
    if sp_score >= 4:
        return build_signal(symbol, "SHORT", "PREMIUM", price, h1, m1,
                             {k: v for k, v in short_premium.items() if v}, sp_score, 5)

    # ════════ KATMAN 2: STANDART (15M + 5M + BTC) ═══════════════
    long_standard = {
        "15M RSI düşük":      m15["rsi"] < cfg["long_rsi_standard"],
        "5M EMA kesişim":     m5["ema_fast"] > m5["ema_slow"] and m5p["ema_fast"] <= m5p["ema_slow"],
        "5M MACD dönüş":      m5["macd_hist"] > 0 and m5p["macd_hist"] <= 0,
        "Hacim 1.6x+":        m5["vol_ratio"] >= 1.6,
        "BTC engel yok":      btc["rsi"] > 35,
    }
    short_standard = {
        "15M RSI yüksek":     m15["rsi"] > cfg["short_rsi_standard"],
        "5M EMA kesişim":     m5["ema_fast"] < m5["ema_slow"] and m5p["ema_fast"] >= m5p["ema_slow"],
        "5M MACD dönüş":      m5["macd_hist"] < 0 and m5p["macd_hist"] >= 0,
        "Hacim 1.6x+":        m5["vol_ratio"] >= 1.6,
        "BTC de zayıf":       btc["rsi"] > 50 or btc["ema_fast"] < btc["ema_slow"],
    }
    ls_score, ss_score = sum(long_standard.values()), sum(short_standard.values())

    if ls_score >= 3:
        return build_signal(symbol, "LONG", "STANDART", price, m15, m1,
                             {k: v for k, v in long_standard.items() if v}, ls_score, 4)
    if ss_score >= 3:
        return build_signal(symbol, "SHORT", "STANDART", price, m15, m1,
                             {k: v for k, v in short_standard.items() if v}, ss_score, 4)

    # ════════ KATMAN 3: HIZLI SCALP (5M + 1M) ═══════════════════
    # BTC ters yöndeyse bu katman tamamen engellenir (risk yönetimi)
    btc_neutral_or_supportive_long  = btc["rsi"] > 40
    btc_neutral_or_supportive_short = btc["rsi"] < 60

    long_scalp = {
        "5M MACD dönüş":      m5["macd_hist"] > 0 and m5p["macd_hist"] <= 0,
        "1M hacim patlama":   m1["vol_ratio"] >= 3.0,
        "1M yeşil mum":       m1["is_green"],
        "Stoch dönüş":        m1["stoch_k"] < 30 and m1["stoch_k"] > m1["stoch_d"],
    }
    short_scalp = {
        "5M MACD dönüş":      m5["macd_hist"] < 0 and m5p["macd_hist"] >= 0,
        "1M hacim patlama":   m1["vol_ratio"] >= 3.0,
        "1M kırmızı mum":     not m1["is_green"],
        "Stoch dönüş":        m1["stoch_k"] > 70 and m1["stoch_k"] < m1["stoch_d"],
    }
    lsc_score, ssc_score = sum(long_scalp.values()), sum(short_scalp.values())

    if lsc_score >= 3 and btc_neutral_or_supportive_long:
        return build_signal(symbol, "LONG", "SCALP", price, m5, m1,
                             {k: v for k, v in long_scalp.items() if v}, lsc_score, 4)
    if ssc_score >= 3 and btc_neutral_or_supportive_short:
        return build_signal(symbol, "SHORT", "SCALP", price, m5, m1,
                             {k: v for k, v in short_scalp.items() if v}, ssc_score, 4)

    return None


def build_signal(symbol, sig_type, tier, price, ref_row, micro_row, conds, score, max_score) -> dict:
    """SL/TP/kaldıraç hesabıyla birlikte sinyal objesi oluşturur."""
    atr = ref_row["atr"]
    bb_low, bb_up = ref_row["bb_low"], ref_row["bb_up"]

    if sig_type == "LONG":
        sl  = bb_low - atr if tier != "SCALP" else price - atr * 1.2
        tp1 = price + atr * 2
        tp2 = bb_up if tier != "SCALP" else price + atr * 3.5
    else:
        sl  = bb_up + atr if tier != "SCALP" else price + atr * 1.2
        tp1 = price - atr * 2
        tp2 = bb_low if tier != "SCALP" else price - atr * 3.5

    risk   = abs(price - sl)
    reward = abs(tp1 - price)
    rr     = reward / risk if risk > 0 else 0

    # Kaldıraç — katman ve R/R'a göre, PARA KAYBETMEME öncelikli
    if tier == "PREMIUM" and rr >= 2.5:
        leverage, risk_label = "x8-10", "💎 ÇOK GÜÇLÜ"
    elif tier == "PREMIUM" and rr >= 1.8:
        leverage, risk_label = "x7", "⭐⭐⭐ GÜÇLÜ"
    elif tier == "STANDART" and rr >= 2.0:
        leverage, risk_label = "x5-6", "⭐⭐ ORTA-GÜÇLÜ"
    elif tier == "STANDART":
        leverage, risk_label = "x4-5", "⭐ ORTA"
    elif tier == "SCALP" and rr >= 1.5:
        leverage, risk_label = "x3-4", "⚡ HIZLI (düşük güven)"
    else:
        leverage, risk_label = "x2-3 (minimum)", "⚡ ZAYIF — dikkatli ol"

    return {
        "symbol": symbol, "type": sig_type, "tier": tier,
        "price": price, "sl": sl, "tp1": tp1, "tp2": tp2,
        "rr": rr, "score": score, "max_score": max_score,
        "leverage": leverage, "risk_label": risk_label,
        "conds": conds, "atr": atr,
    }


# ─────────────────────────────────────────────────────────────
#  RİSK YÖNETİMİ — para kaybetmeme kontrolü
# ─────────────────────────────────────────────────────────────

def reset_daily_tracker_if_needed():
    today = datetime.now().date()
    if daily_pnl_tracker["date"] != today:
        daily_pnl_tracker["date"] = today
        daily_pnl_tracker["loss_pct"] = 0.0
        daily_pnl_tracker["consecutive_losses"] = 0


def is_trading_paused() -> str | None:
    """Günlük limit veya ardışık kayıp varsa işlem önerilerini durdurur."""
    reset_daily_tracker_if_needed()
    if daily_pnl_tracker["loss_pct"] >= DAILY_MAX_LOSS_PCT:
        return f"🛑 Günlük kayıp limiti (%{DAILY_MAX_LOSS_PCT}) doldu — bugün yeni işlem önerilmiyor."
    if daily_pnl_tracker["consecutive_losses"] >= CONSECUTIVE_LOSS_PAUSE:
        return f"🛑 {CONSECUTIVE_LOSS_PAUSE} ardışık kayıp algılandı — 1 saat ara verilmesi öneriliyor."
    return None


def is_duplicate(symbol: str, sig: dict) -> bool:
    if symbol not in last_signals:
        return False
    prev = last_signals[symbol]
    if prev["type"] != sig["type"]:
        return False
    price_diff = abs(sig["price"] - prev["price"]) / prev["price"]
    return price_diff < 0.004


# ─────────────────────────────────────────────────────────────
#  MESAJ FORMATI
# ─────────────────────────────────────────────────────────────

TIER_EMOJI = {"PREMIUM": "🥇", "STANDART": "🥈", "SCALP": "🥉"}

def format_message(sig: dict) -> str:
    is_long = sig["type"] == "LONG"
    emoji   = "🟢" if is_long else "🔴"
    p, sl, tp1, tp2 = sig["price"], sig["sl"], sig["tp1"], sig["tp2"]
    sl_pct  = (sl  - p) / p * 100
    tp1_pct = (tp1 - p) / p * 100
    tp2_pct = (tp2 - p) / p * 100
    now     = datetime.now().strftime("%H:%M:%S")
    cond_lines = "\n".join(f"  ✓ {c}" for c in sig["conds"])
    tier_e  = TIER_EMOJI.get(sig["tier"], "")

    risk_warn = ""
    if sig["tier"] == "SCALP":
        risk_warn = "\n⚠️ _Bu hızlı scalp sinyali — düşük kaldıraç kullan, küçük pozisyon._"

    return (
        f"{emoji} {tier_e} *{sig['symbol']} — {sig['type']}* ({sig['tier']})\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Giriş: `{p:.2e}`\n"
        f"🛑 SL:    `{sl:.2e}` ({sl_pct:+.1f}%)\n"
        f"🎯 TP1:   `{tp1:.2e}` ({tp1_pct:+.1f}%)\n"
        f"🎯 TP2:   `{tp2:.2e}` ({tp2_pct:+.1f}%)\n"
        f"📊 R/R:   `{sig['rr']:.1f}:1`\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ Koşullar ({sig['score']}/{sig['max_score']}):\n{cond_lines}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚡ Güç: {sig['risk_label']}\n"
        f"💪 Önerilen kaldıraç: `{sig['leverage']}`\n"
        f"⏰ {now}"
        f"{risk_warn}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ _Finansal tavsiye değildir. Pozisyon riskini sermayenin %{MAX_RISK_PER_TRADE_PCT}'i ile sınırla._"
    )


# ─────────────────────────────────────────────────────────────
#  ANA TARAMA
# ─────────────────────────────────────────────────────────────

async def scan(bot: Bot) -> None:
    pause_reason = is_trading_paused()
    if pause_reason:
        log.warning(pause_reason)
        return   # Risk limiti doldu, sinyal arama

    try:
        btc_15m = add_indicators(fetch_ohlcv(BTC_SYMBOL, "15m", limit=60))

        for symbol in COINS:
            try:
                dfs = {
                    "1m":  add_indicators(fetch_ohlcv(symbol, "1m",  limit=60)),
                    "5m":  add_indicators(fetch_ohlcv(symbol, "5m",  limit=60)),
                    "15m": add_indicators(fetch_ohlcv(symbol, "15m", limit=60)),
                    "1h":  add_indicators(fetch_ohlcv(symbol, "1h",  limit=60)),
                }
                sig = evaluate_tiered(symbol, dfs, btc_15m)

                if sig and not is_duplicate(symbol, sig):
                    msg = format_message(sig)
                    await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
                    last_signals[symbol] = {"type": sig["type"], "price": sig["price"], "time": datetime.now()}
                    log.info("Sinyal: %s %s [%s] @ %.2e (R/R=%.1f)", symbol, sig["type"], sig["tier"], sig["price"], sig["rr"])
                else:
                    log.debug("Sinyal yok: %s", symbol)

            except Exception as e:
                log.error("Hata (%s): %s", symbol, e)

            await asyncio.sleep(0.3)

    except Exception as e:
        log.error("Genel tarama hatası: %s", e)


async def background_loop(bot: Bot) -> None:
    while True:
        await scan(bot)
        await asyncio.sleep(CHECK_INTERVAL)


# ─────────────────────────────────────────────────────────────
#  TELEGRAM KOMUTLARI
# ─────────────────────────────────────────────────────────────

async def cmd_start(update, context):
    await update.message.reply_text(
        "🤖 *Katmanlı Sinyal Botu aktif!*\n\n"
        "🥇 PREMIUM — günde ~1-2, en güvenilir\n"
        "🥈 STANDART — günde ~3-5, orta güven\n"
        "🥉 SCALP — günde ~8-15, hızlı/düşük güven\n\n"
        "/durum — Tüm coinlerin anlık durumu\n"
        "/risk — Günlük risk durumu\n"
        "/kayıp [yüzde] — Manuel kayıp kaydet (örn: /kayıp 1.5)\n"
        "/sıfırla — Günlük sayaçları sıfırla",
        parse_mode="Markdown",
    )


async def cmd_durum(update, context):
    try:
        btc_15m = add_indicators(fetch_ohlcv(BTC_SYMBOL, "15m", limit=60))
        b = btc_15m.iloc[-1]
        lines = [f"📊 *Genel Durum* — {datetime.now().strftime('%H:%M')}\n"]
        lines.append(f"BTC RSI(15M): `{b['rsi']:.0f}` | Trend: `{'YUKARI' if b['ema_fast']>b['ema_slow'] else 'ASAGI'}`\n")

        for symbol in COINS:
            h1 = add_indicators(fetch_ohlcv(symbol, "1h", limit=60)).iloc[-1]
            m15 = add_indicators(fetch_ohlcv(symbol, "15m", limit=60)).iloc[-1]
            lines.append(
                f"\n*{symbol}*\n"
                f"  Fiyat: `{h1['close']:.2e}`\n"
                f"  RSI 1H/15M: `{h1['rsi']:.0f}` / `{m15['rsi']:.0f}`\n"
                f"  Hacim: `{h1['vol_ratio']:.1f}x`"
            )
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Hata: {e}")


async def cmd_risk(update, context):
    reset_daily_tracker_if_needed()
    pause = is_trading_paused()
    status = pause if pause else "✅ Normal — işlem önerilerine devam ediliyor"
    await update.message.reply_text(
        f"📋 *Risk Durumu*\n"
        f"Günlük kayıp: `%{daily_pnl_tracker['loss_pct']:.1f}` / %{DAILY_MAX_LOSS_PCT}\n"
        f"Ardışık kayıp: `{daily_pnl_tracker['consecutive_losses']}` / {CONSECUTIVE_LOSS_PAUSE}\n\n"
        f"{status}",
        parse_mode="Markdown",
    )


async def cmd_kayip(update, context):
    """Kullanıcı manuel olarak kaybını bildirir, bot risk sayaçlarını günceller."""
    try:
        pct = float(context.args[0])
        daily_pnl_tracker["loss_pct"] += abs(pct)
        daily_pnl_tracker["consecutive_losses"] += 1
        await update.message.reply_text(
            f"📉 Kayıp kaydedildi: %{pct}\n"
            f"Toplam günlük kayıp: %{daily_pnl_tracker['loss_pct']:.1f}\n"
            f"Ardışık kayıp: {daily_pnl_tracker['consecutive_losses']}"
        )
        pause = is_trading_paused()
        if pause:
            await update.message.reply_text(pause)
    except (IndexError, ValueError):
        await update.message.reply_text("Kullanım: /kayıp 1.5  (yüzde olarak)")


async def cmd_sifirla(update, context):
    daily_pnl_tracker["loss_pct"] = 0.0
    daily_pnl_tracker["consecutive_losses"] = 0
    await update.message.reply_text("✅ Günlük sayaçlar sıfırlandı.")


async def post_init(application):
    asyncio.create_task(background_loop(application.bot))
    try:
        await application.bot.send_message(
            chat_id=CHAT_ID,
            text=(
                "✅ *Katmanlı Sinyal Botu başlatıldı!*\n"
                f"⏱ Kontrol: her {CHECK_INTERVAL}s\n"
                f"🪙 Takip: {', '.join(COINS.keys())}\n"
                f"🛡 Risk limiti: max %{MAX_RISK_PER_TRADE_PCT}/işlem, %{DAILY_MAX_LOSS_PCT}/gün\n\n"
                "/start yazarak komutları gör."
            ),
            parse_mode="Markdown",
        )
    except Exception as e:
        log.warning("Başlangıç mesajı gönderilemedi: %s", e)


def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("durum",    cmd_durum))
    app.add_handler(CommandHandler("risk",     cmd_risk))
    app.add_handler(CommandHandler("kayıp",    cmd_kayip))
    app.add_handler(CommandHandler("sifirla",  cmd_sifirla))
    log.info("Bot başlatıldı.")
    app.run_polling()


if __name__ == "__main__":
    main()
