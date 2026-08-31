/*
 * test_harness.cpp - Phase 2 pyfofem C++ oracle harness.
 *
 * Implements the shared strict harness contract and six scientific modes
 * from development/plans/gate0/05-harness-contract.md:
 *   consume, litter_eq, shrub_herb_eq, mortality, bark_thick, canopy_cover
 * (soil_campbell is Phase 5's; the removed emissions_state mode is not
 * reintroduced here — see gate0/05-harness-contract.md §8).
 *
 * Usage:
 *   fofem_test <input.csv> <output_prefix> [--species-csv <path>]
 *
 * --species-csv is REQUIRED for mortality, bark_thick, and canopy_cover
 * (species table load + startup qualification happens before any row is
 * processed — see load_species_table()) and REJECTED for consume,
 * litter_eq, and shrub_herb_eq (those modes never touch the species table).
 *
 * Input file format (every mode):
 *   line 1: #fofem-harness,<mode>,<schema_version>
 *   line 2: exact column header for that mode
 *   line 3+: data rows, comma-separated, case_id first, expect_error second
 *
 * Every mode's outputs use the four output roles defined in the contract's
 * §1 (PRIMARY / SECONDARY FAN-OUT / SECONDARY SCIENTIFIC AGGREGATE /
 * DIAGNOSTIC GROUP STATUS). See per-mode run_* functions below for exact
 * schemas, each with its gate0/05-harness-contract.md section cited.
 *
 * Oracle independence: every mode calls the real pinned FOFEM function(s)
 * it names. No equation is reimplemented in this file.
 *
 * Gate 0 contract correction (Phase 2, direct C++ evidence, Codex-approved):
 * the originally approved contract specified MRT_InitST() to populate the
 * species table for mortality/bark_thick/canopy_cover. Direct evidence
 * (fof_spp.h:10-19; see load_species_table() below) shows MRT_InitST()'s
 * table uses obsolete species codes real FOFEM does not use for this
 * purpose. These three modes now require an explicit, real species CSV
 * (the tracked src/pyfofem/supporting_data/FOFEM6.7/FOF_SPP.CSV) loaded via
 * the real production entry point MRT_LoadSpe(). Gate 0 correctly
 * identified the need to initialize the species table; Phase 2 corrected
 * which initializer is oracle-faithful.
 */

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
#include "fof_ansi.h"
#include "fof_ci.h"
#include "fof_co.h"
#include "fof_cm.h"
#include "fof_co2.h"
#include "fof_duf.h"
#include "fof_hsf.h"
#include "fof_mec.h"
#include "fof_mis.h"
#include "fof_mrt.h"
#include "fof_nes.h"
#include "fof_pf2.h"
#include "fof_sgv.h"
#include "fof_sh.h"
#include "fof_smt.h"
}

// fof_cct.h bundles CCT_Get()'s declaration together with a DEFINITION
// (not just a declaration) of the sr_CCT[] global (fof_cct.h:49). It is
// designed to be included exactly once, inside fof_mrt.cpp; including it
// here too would duplicate that global against the copy fof_mrt.cpp
// already provides and fail to link (confirmed: LNK2005 sr_CCT already
// defined). Redeclare only the ABI-compatible pieces this harness needs —
// struct layout and signature verified field-for-field against
// fof_cct.h:38-44/90.
extern "C" {
typedef struct {
  int i_No;
  char cr_CC[10];
  float f_a;
  float f_b;
  float f_r;
} d_CCT;
int CCT_Get(int i_No, d_CCT *a_CCT);
}

// ===========================================================================
// SHA-256 (self-contained; no external dependency). Standard FIPS 180-4
// algorithm, byte-for-byte per the published constants/spec.
// ===========================================================================
namespace sha256_impl {

typedef uint32_t u32;
typedef uint64_t u64;

static const u32 K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

static inline u32 rotr(u32 x, u32 n) { return (x >> n) | (x << (32 - n)); }

struct Ctx {
  u32 h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
              0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  u64 total_len = 0;
  std::vector<unsigned char> buffer;

  void transform(const unsigned char *chunk) {
    u32 w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (u32(chunk[i * 4]) << 24) | (u32(chunk[i * 4 + 1]) << 16) |
             (u32(chunk[i * 4 + 2]) << 8) | u32(chunk[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      u32 s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      u32 s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    u32 a = h[0], b = h[1], c = h[2], d = h[3];
    u32 e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      u32 S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      u32 ch = (e & f) ^ ((~e) & g);
      u32 temp1 = hh + S1 + ch + K[i] + w[i];
      u32 S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      u32 maj = (a & b) ^ (a & c) ^ (b & c);
      u32 temp2 = S0 + maj;
      hh = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }

  void update(const unsigned char *data, size_t len) {
    total_len += len;
    buffer.insert(buffer.end(), data, data + len);
    size_t i = 0;
    while (buffer.size() - i >= 64) {
      transform(&buffer[i]);
      i += 64;
    }
    buffer.erase(buffer.begin(), buffer.begin() + i);
  }

  std::string hexdigest() {
    u64 bit_len = total_len * 8;
    buffer.push_back(0x80);
    while (buffer.size() % 64 != 56) buffer.push_back(0x00);
    for (int i = 7; i >= 0; --i)
      buffer.push_back((unsigned char)((bit_len >> (i * 8)) & 0xff));
    for (size_t i = 0; i < buffer.size(); i += 64) transform(&buffer[i]);
    static const char *hexch = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (int i = 0; i < 8; ++i) {
      for (int j = 3; j >= 0; --j) {
        unsigned char byte = (unsigned char)((h[i] >> (j * 8)) & 0xff);
        out.push_back(hexch[byte >> 4]);
        out.push_back(hexch[byte & 0xf]);
      }
    }
    return out;
  }
};

}  // namespace sha256_impl

static std::string sha256_hex(const std::string &data) {
  sha256_impl::Ctx ctx;
  ctx.update(reinterpret_cast<const unsigned char *>(data.data()),
             data.size());
  return ctx.hexdigest();
}

static std::string sha256_hex_file(const std::string &path, bool *ok) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    *ok = false;
    return "";
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  *ok = true;
  return sha256_hex(ss.str());
}

// ===========================================================================
// Small string utilities
// ===========================================================================

static std::string trim(const std::string &s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace((unsigned char)s[a])) ++a;
  while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
  return s.substr(a, b - a);
}

static std::vector<std::string> split_comma(const std::string &line) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : line) {
    if (c == ',') {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

// CSV-quote a field for OUTPUT only (err_text may contain commas/quotes from
// C++ error messages). Contract: "quoted and newline-escaped".
static std::string csv_quote(const std::string &field) {
  bool needs_quote = field.find_first_of(",\"\n\r") != std::string::npos;
  std::string s = field;
  // Escape embedded newlines/carriage returns as literal \n / \r so the
  // output stays one physical line per row.
  std::string escaped;
  for (char c : s) {
    if (c == '\n') escaped += "\\n";
    else if (c == '\r') escaped += "\\r";
    else escaped += c;
  }
  if (!needs_quote && escaped.find('\\') == std::string::npos) return escaped;
  std::string out = "\"";
  for (char c : escaped) {
    if (c == '"') out += "\"\"";
    else out += c;
  }
  out += "\"";
  return out;
}

// Bounded copy into a fixed-size char buffer without any CRT function MSVC
// flags "unsafe" (strncpy/strcpy/strcat) — keeps the harness warning-clean
// without disabling secure-CRT diagnostics for the target or touching
// pinned upstream source.
// Returns false (dst left untouched) if src does not fit — a CSV-derived
// string is NEVER silently truncated to fit a C buffer. Callers copying
// user/CSV-derived data must check the return value and fail closed (audit
// finding #3).
template <size_t N>
static bool safe_copy(char (&dst)[N], const std::string &src) {
  if (src.size() > N - 1) return false;
  std::memcpy(dst, src.data(), src.size());
  dst[src.size()] = '\0';
  return true;
}

// For copying a value the harness itself controls (a fixed macro literal
// or an empty string) into a buffer already known by inspection to be large
// enough — never used for CSV-derived data. Asserts rather than silently
// truncating if that invariant is ever violated by a future edit.
template <size_t N>
static void safe_copy_literal(char (&dst)[N], const std::string &src) {
  bool ok = safe_copy(dst, src);
  assert(ok && "safe_copy_literal: internal literal does not fit destination buffer");
  (void)ok;
}

// ===========================================================================
// Strict field parsers (harness contract §1 "Parsing")
// ===========================================================================

// Strict double: full-consumption strtod, no blank, no nan/inf, no hex float.
static bool parse_strict_double(const std::string &raw, double *out,
                                 std::string *err) {
  std::string s = trim(raw);
  if (s.empty()) {
    *err = "blank numeric field";
    return false;
  }
  std::string lower = s;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                  [](unsigned char c) { return (char)std::tolower(c); });
  if (lower.find("nan") != std::string::npos ||
      lower.find("inf") != std::string::npos) {
    *err = "nan/inf not permitted";
    return false;
  }
  if (lower.find("0x") != std::string::npos) {
    *err = "hex float not permitted";
    return false;
  }
  const char *cs = s.c_str();
  char *endp = nullptr;
  errno = 0;
  double v = std::strtod(cs, &endp);
  int strtod_errno = errno;
  if (endp != cs + s.size()) {
    *err = "trailing junk after numeric field";
    return false;
  }
  if (endp == cs) {
    *err = "no digits parsed";
    return false;
  }
  // strtod sets ERANGE on overflow (returns +/-HUGE_VAL) and, per the C
  // standard and MSVC's documented behaviour, also on severe underflow
  // (returns a value <= the smallest normalized nonzero double, including
  // exactly 0.0). Reject both explicitly rather than accepting a silently
  // clamped/flushed magnitude.
  if (strtod_errno == ERANGE) {
    *err = "numeric field out of representable double range "
           "(overflow or underflow)";
    return false;
  }
  // Backstop independent of errno/ERANGE portability: never accept a
  // non-finite result merely because the source text did not literally
  // contain "inf" (e.g. "1e999" has no such substring but overflows).
  if (!std::isfinite(v)) {
    *err = "numeric field parsed to a non-finite value";
    return false;
  }
  *out = v;
  return true;
}

// case_id: [A-Za-z0-9_.-]{1,64}
static bool parse_case_id(const std::string &raw, std::string *err) {
  if (raw.empty() || raw.size() > 64) {
    *err = "case_id length out of [1,64]";
    return false;
  }
  for (char c : raw) {
    bool ok = std::isalnum((unsigned char)c) || c == '_' || c == '.' ||
              c == '-';
    if (!ok) {
      *err = "case_id contains a disallowed character";
      return false;
    }
  }
  return true;
}

// expect_error: exactly "0" or "1"
static bool parse_expect_error(const std::string &raw, bool *out,
                                std::string *err) {
  if (raw == "0") { *out = false; return true; }
  if (raw == "1") { *out = true; return true; }
  *err = "expect_error must be exactly 0 or 1";
  return false;
}

// emission-factor group id: canonical single digit "1".."8" (no sign, no
// leading zero, no decimal point, no whitespace). See harness-contract §2.
static bool parse_group_id(const std::string &raw, std::string *canonical,
                            std::string *err) {
  if (raw.size() != 1 || raw[0] < '1' || raw[0] > '8') {
    *err = "group id must be a single canonical digit 1-8";
    return false;
  }
  *canonical = raw;
  return true;
}

// i_Eq* batch-override columns: integer, "-1" sentinel for "not set".
static bool parse_eq_override(const std::string &raw, int *out,
                               std::string *err) {
  std::string s = trim(raw);
  if (s.empty()) { *err = "blank eq override field"; return false; }
  const char *cs = s.c_str();
  char *endp = nullptr;
  errno = 0;
  long v = std::strtol(cs, &endp, 10);
  int strtol_errno = errno;
  if (endp != cs + s.size() || endp == cs) {
    *err = "eq override must be a plain integer";
    return false;
  }
  if (strtol_errno == ERANGE) {
    *err = "eq override integer out of representable long range";
    return false;
  }
  if (v < static_cast<long>(std::numeric_limits<int>::min()) ||
      v > static_cast<long>(std::numeric_limits<int>::max())) {
    *err = "eq override integer out of int range";
    return false;
  }
  *out = static_cast<int>(v);
  return true;
}

// ===========================================================================
// Row/outcome bookkeeping (harness contract §1 "Execution" / "Outcome
// policy")
// ===========================================================================

enum class Outcome { OK, EXPECTED_MODEL_ERROR, UNEXPECTED_FAILURE };

static const char *outcome_str(Outcome o) {
  switch (o) {
    case Outcome::OK: return "ok";
    case Outcome::EXPECTED_MODEL_ERROR: return "expected_model_error";
    default: return "unexpected_failure";
  }
}

// Two-sided expect_error classification (harness contract §1).
static Outcome classify(bool expect_error, bool model_errored) {
  if (expect_error) {
    return model_errored ? Outcome::EXPECTED_MODEL_ERROR
                          : Outcome::UNEXPECTED_FAILURE;
  }
  return model_errored ? Outcome::UNEXPECTED_FAILURE : Outcome::OK;
}

// ===========================================================================
// Input file reading: magic/version line + header validation
// ===========================================================================

struct InputFile {
  std::string mode;
  std::string schema_version;
  std::vector<std::string> header;
  std::vector<std::vector<std::string>> rows;  // raw fields, one vec per row
  std::vector<std::string> row_hashes;         // input_sha256 per row
};

static bool read_input_file(const std::string &path,
                             const std::vector<std::string> &expected_header,
                             InputFile *out, std::string *err) {
  std::ifstream f(path);
  if (!f) {
    *err = "cannot open input file: " + path;
    return false;
  }
  std::string magic_line;
  if (!std::getline(f, magic_line)) {
    *err = "empty input file (no magic/version line)";
    return false;
  }
  // strip CR if present (CRLF files)
  if (!magic_line.empty() && magic_line.back() == '\r') magic_line.pop_back();
  std::vector<std::string> magic_fields = split_comma(magic_line);
  if (magic_fields.size() != 3 || magic_fields[0] != "#fofem-harness") {
    *err = "malformed magic/version line: " + magic_line;
    return false;
  }
  out->mode = magic_fields[1];
  out->schema_version = magic_fields[2];

  std::string header_line;
  if (!std::getline(f, header_line)) {
    *err = "missing header line (file has only a magic line)";
    return false;
  }
  if (!header_line.empty() && header_line.back() == '\r') header_line.pop_back();
  out->header = split_comma(header_line);
  if (out->header.size() != expected_header.size()) {
    *err = "header field count mismatch: got " +
           std::to_string(out->header.size()) + ", expected " +
           std::to_string(expected_header.size());
    return false;
  }
  for (size_t i = 0; i < expected_header.size(); ++i) {
    std::string got = trim(out->header[i]);
    if (got != expected_header[i]) {
      *err = "header column " + std::to_string(i) + " is '" + got +
             "', expected '" + expected_header[i] + "'";
      return false;
    }
  }

  std::set<std::string> seen_ids;
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    std::vector<std::string> fields = split_comma(line);
    if (fields.size() != expected_header.size()) {
      *err = "row field count mismatch on row " +
             std::to_string(out->rows.size() + 1) + ": got " +
             std::to_string(fields.size()) + ", expected " +
             std::to_string(expected_header.size());
      return false;
    }
    std::string cid_err;
    if (!parse_case_id(fields[0], &cid_err)) {
      *err = "row " + std::to_string(out->rows.size() + 1) +
             " case_id invalid: " + cid_err;
      return false;
    }
    if (seen_ids.count(fields[0])) {
      *err = "duplicate case_id: " + fields[0];
      return false;
    }
    seen_ids.insert(fields[0]);
    // Normalize (trim) every field EXACTLY ONCE, then use that same
    // trimmed vector for both execution (stored into out->rows, which
    // every run_* function reads and copies/parses from) and input_sha256.
    // Previously the hash was computed from a locally trimmed copy while
    // out->rows kept the raw (untrimmed) fields — a string field with
    // incidental leading/trailing whitespace would then execute
    // differently (e.g. safe_copy'd as-is into a fixed struct buffer,
    // changing C++-side string comparisons) than what the hash certified,
    // so two genuinely different executed inputs could share one
    // input_sha256. Trimming once here closes that gap: hash and
    // execution now always see the identical representation.
    std::vector<std::string> normalized_fields;
    normalized_fields.reserve(fields.size());
    std::string normalised;
    for (size_t i = 0; i < fields.size(); ++i) {
      std::string t = trim(fields[i]);
      normalized_fields.push_back(t);
      if (i) normalised += ",";
      normalised += t;
    }
    out->rows.push_back(normalized_fields);
    out->row_hashes.push_back(sha256_hex(normalised));
  }
  if (out->rows.empty()) {
    *err = "no data rows (header-only input) — at least one data row is required";
    return false;
  }
  return true;
}

// ===========================================================================
// Output file writer helper
// ===========================================================================

// Fails closed on: a row whose width differs from the declared header, a
// stream write failure, or a flush/close failure. Every run_* function
// must check `.failed` (set by any of those three) before its final
// return, in addition to calling `.close_and_check()` once after its last
// row — a caller that ignores `.failed` would silently accept a
// truncated/malformed output file as a successful golden.
struct CsvWriter {
  std::ofstream f;
  bool ok = false;
  bool failed = false;
  size_t expected_width = 0;
  bool width_set = false;

  explicit CsvWriter(const std::string &path) {
    f.open(path, std::ios::binary);
    ok = (bool)f;
    if (!ok) failed = true;
  }

  void header(const std::vector<std::string> &cols) {
    expected_width = cols.size();
    width_set = true;
    write_line(cols);
  }

  void row(const std::vector<std::string> &vals) {
    if (width_set && vals.size() != expected_width) {
      failed = true;
      return;
    }
    write_line(vals);
  }

  // Must be called once, after the last row, before treating the file as
  // successfully produced.
  bool close_and_check() {
    f.flush();
    if (!f) failed = true;
    f.close();
    if (f.fail()) failed = true;
    return !failed;
  }

 private:
  void write_line(const std::vector<std::string> &vals) {
    for (size_t i = 0; i < vals.size(); ++i) {
      if (i) f << ",";
      f << vals[i];
    }
    f << "\n";
    if (!f) failed = true;
  }
};

// Set by fmt() whenever it is asked to format a non-finite value on a
// declared-successful ("ok") row. Every run_* function resets this to
// false at the top of its per-row loop and checks it before writing that
// row's outcome as "ok" — a real successful computation should never
// legitimately produce NaN/Inf, and the harness must fail closed rather
// than silently emit "nan"/"inf" text (not the documented NA sentinel)
// into a CSV a downstream comparison would parse as a float.
static bool g_output_nonfinite = false;

template <typename T>
static std::string fmt(T v, int precision = 6) {
  double d = (double)v;
  if (!std::isfinite(d)) {
    g_output_nonfinite = true;
    return "NONFINITE";
  }
  char buf[64];
  snprintf(buf, sizeof(buf), "%.*f", precision, d);
  return std::string(buf);
}

// ===========================================================================
// Mode: consume  (gate0/05-harness-contract.md §2, §2a)
// ===========================================================================

static const std::vector<std::string> CONSUME_HEADER = {
    "case_id", "expect_error",
    "litter_tac", "duff_tac", "duff_depth_in", "duff_moist_pct",
    "herb_tac", "shrub_tac", "crown_fol_tac", "crown_bra_tac", "pct_crown_burn",
    "dw10_moist_pct", "dw1000_moist_pct", "litter_moist_pct",
    "dw1_tac", "dw10_tac", "dw100_tac", "dw1000_tac", "pct_rot",
    "snd_dw3_tac", "snd_dw6_tac", "snd_dw9_tac", "snd_dw20_tac",
    "rot_dw3_tac", "rot_dw6_tac", "rot_dw9_tac", "rot_dw20_tac",
    "region", "season", "fuel_cat", "cover_group", "cover_class",
    "duff_moist_method",
    "intensity_kw_m", "ig_time_s", "windspeed_m_s", "depth_ft",
    "ambient_temp_c",
    "critical_intensity_kw_m", "ef_flame_group", "ef_smolder_group",
    "ef_duff_group",
    "batch_equ", "eq_lit", "eq_duf_loa", "eq_duf_dep", "eq_mse", "eq_herb",
    "eq_shrub"};

static const std::vector<std::string> CONSUME_SUMMARY_META = {
    "case_id", "mode", "schema_version", "outcome", "ret_code", "err_text",
    "input_sha256"};

static const std::vector<std::string> CONSUME_SCIENTIFIC = {
    "LitPre", "LitCon", "LitPos",
    "DW1Pre", "DW1Con", "DW1Pos",
    "DW10Pre", "DW10Con", "DW10Pos",
    "DW100Pre", "DW100Con", "DW100Pos",
    "SndDW1kPre", "SndDW1kCon", "SndDW1kPos",
    "RotDW1kPre", "RotDW1kCon", "RotDW1kPos",
    "DufPre", "DufCon", "DufPos", "DufPer",
    "HerPre", "HerCon", "HerPos",
    "ShrPre", "ShrCon", "ShrPos",
    "FolPre", "FolCon", "FolPos",
    "BraPre", "BraCon", "BraPos",
    "TotPre", "TotCon", "TotPos",
    "FlaCon", "SmoCon", "FlaDur", "SmoDur",
    "PM25F", "PM25S", "PM10F", "PM10S", "CH4F", "CH4S", "COF", "COS",
    "CO2F", "CO2S", "NOXF", "NOXS", "SO2F", "SO2S",
    "PM25S_Duff", "PM10S_Duff", "CH4S_Duff", "COS_Duff", "CO2S_Duff",
    "NOXS_Duff", "SO2S_Duff",
    "MSE", "DufDepPre", "DufDepCon", "DufDepPos"};

static const std::vector<std::string> CONSUME_COMPONENTS_HEADER = {
    "case_id", "component", "pre_tac", "con_tac", "pos_tac", "pct_con",
    "equation", "input_sha256"};

static const std::vector<std::string> CONSUME_COMPONENT_NAMES = {
    "Litter", "DW1", "DW10", "DW100", "SndDW1k", "RotDW1k", "Duff", "Herb",
    "Shrub", "Foliage", "Branch"};

// Sentinel for "not applicable" free-text/override fields the mode allows to
// be absent. Documented once here; applied uniformly across every mode in
// this file (Phase 2 implementation decision — the plan authorises a
// per-column sentinel without pinning its literal spelling).
static const char *NA_SENTINEL = "NA";

static bool nes_read_once(std::string *factor_table_sha256_out) {
  char err_buf[3000];
  err_buf[0] = '\0';
  int ret = NES_Read((char *)"", err_buf);
  bool hash_ok = false;
  std::string h = sha256_hex_file("Emission_Factors.csv", &hash_ok);
  if (hash_ok) *factor_table_sha256_out = h;
  if (ret != 1) {
    std::cerr << "[fofem_test] FATAL: NES_Read failed (ret=" << ret
              << "): " << err_buf << "\n";
    return false;
  }
  if (!hash_ok) {
    std::cerr << "[fofem_test] FATAL: could not hash Emission_Factors.csv "
                 "for provenance (CWD must be FOF_UNIX/)\n";
    return false;
  }
  return true;
}

// Load one 7-field emission-factor block (harness-contract §2a step 5).
// Returns false (fatal) if the lookup fails or the block is all-zero.
static bool load_factor_block(const std::string &group_canonical,
                               float *CO, float *CO2, float *CH4,
                               float *PM25, float *PM10, float *NOX,
                               float *SO2, std::string *err) {
  char grp[8];
  snprintf(grp, sizeof(grp), "%s", group_canonical.c_str());
  *CO = *CO2 = *CH4 = *PM25 = *PM10 = *NOX = *SO2 = 0.0f;
  int ret = NES_Get_MajFactor(grp, CO, CO2, CH4, PM25, PM10, NOX, SO2);
  if (ret != 1) {
    *err = "NES_Get_MajFactor failed for group '" + group_canonical + "'";
    return false;
  }
  if (*CO == 0.0f && *CO2 == 0.0f && *CH4 == 0.0f && *PM25 == 0.0f &&
      *PM10 == 0.0f && *NOX == 0.0f && *SO2 == 0.0f) {
    *err = "factor block for group '" + group_canonical +
           "' is wholly zero (gate 4 backstop)";
    return false;
  }
  return true;
}

static int run_consume(const InputFile &in, const std::string &prefix) {
  std::string factor_table_sha256;
  if (!nes_read_once(&factor_table_sha256)) return 1;
  std::cout << "FACTOR_TABLE_SHA256=" << factor_table_sha256 << "\n";

  CsvWriter summary(prefix + "_summary.csv");
  CsvWriter components(prefix + "_components.csv");
  if (!summary.ok || !components.ok) {
    std::cerr << "[fofem_test] FATAL: cannot open output files\n";
    return 1;
  }
  std::vector<std::string> summary_header = CONSUME_SUMMARY_META;
  summary_header.insert(summary_header.end(), CONSUME_SCIENTIFIC.begin(),
                         CONSUME_SCIENTIFIC.end());
  summary.header(summary_header);
  components.header(CONSUME_COMPONENTS_HEADER);

  bool any_unexpected = false;
  size_t ok_rows = 0;
  size_t components_rows_written = 0;

  for (size_t r = 0; r < in.rows.size(); ++r) {
    const auto &f = in.rows[r];
    g_output_nonfinite = false;  // reset per row; fmt() sets this if called
    std::string row_err;
    bool expect_error = false;
    if (!parse_expect_error(f[1], &expect_error, &row_err)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1 << ": " << row_err << "\n";
      return 1;
    }

    // d_CI/d_CO are large pinned struct types (2900 + 240632 bytes) that
    // previously lived on this function's own stack frame, which /analyze
    // flags as C6262 ("Function uses N bytes of stack") — a real, honest
    // measurement of these two structs' combined size, not a /RTC1
    // analysis artifact. Heap-allocating them here (harness-owned code,
    // not the pinned scientific structs/functions themselves) removes the
    // large-stack warning at the source; every `ci.`/`co.` access below is
    // unchanged since both are still plain d_CI&/d_CO& references.
    auto ci_storage = std::make_unique<d_CI>();
    auto co_storage = std::make_unique<d_CO>();
    d_CI &ci = *ci_storage;
    d_CO &co = *co_storage;
    CI_Init(&ci);
    CO_Init(&co);

    double dv;
#define REQ_DOUBLE(idx, target)                                            \
  if (!parse_strict_double(f[idx], &dv, &row_err)) {                       \
    std::cerr << "[fofem_test] FATAL row " << r + 1 << " field '"          \
              << CONSUME_HEADER[idx] << "': " << row_err << "\n";          \
    return 1;                                                              \
  }                                                                         \
  (target) = (float)dv;

    REQ_DOUBLE(2, ci.f_Lit);
    REQ_DOUBLE(3, ci.f_Duff);
    REQ_DOUBLE(4, ci.f_DufDep);
    REQ_DOUBLE(5, ci.f_MoistDuff);
    REQ_DOUBLE(6, ci.f_Herb);
    REQ_DOUBLE(7, ci.f_Shrub);
    REQ_DOUBLE(8, ci.f_CroFol);
    REQ_DOUBLE(9, ci.f_CroBra);
    REQ_DOUBLE(10, ci.f_Pc_CroBrn);
    REQ_DOUBLE(11, ci.f_MoistDW10);
    REQ_DOUBLE(12, ci.f_MoistDW1000);
    REQ_DOUBLE(13, ci.f_LitMoi);
    REQ_DOUBLE(14, ci.f_DW1);
    REQ_DOUBLE(15, ci.f_DW10);
    REQ_DOUBLE(16, ci.f_DW100);
    REQ_DOUBLE(17, ci.f_DW1000);
    REQ_DOUBLE(18, ci.f_pcRot);
    REQ_DOUBLE(19, ci.f_Snd_DW3);
    REQ_DOUBLE(20, ci.f_Snd_DW6);
    REQ_DOUBLE(21, ci.f_Snd_DW9);
    REQ_DOUBLE(22, ci.f_Snd_DW20);
    REQ_DOUBLE(23, ci.f_Rot_DW3);
    REQ_DOUBLE(24, ci.f_Rot_DW6);
    REQ_DOUBLE(25, ci.f_Rot_DW9);
    REQ_DOUBLE(26, ci.f_Rot_DW20);

#define REQ_STR(idx, target)                                                \
  if (!safe_copy((target), f[idx])) {                                      \
    std::cerr << "[fofem_test] FATAL row " << r + 1 << " field '"          \
              << CONSUME_HEADER[idx] << "': value exceeds destination "    \
                 "buffer capacity (" << (sizeof(target) - 1)               \
              << " chars) — not truncated\n";                              \
    return 1;                                                              \
  }
    REQ_STR(27, ci.cr_Region);
    REQ_STR(28, ci.cr_Season);
    REQ_STR(29, ci.cr_FuelCategory);
    REQ_STR(30, ci.cr_CoverGroup);
    REQ_STR(31, ci.cr_CoverClass);
    REQ_STR(32, ci.cr_DufMoiMet);

    REQ_DOUBLE(33, ci.f_INTENSITY);
    REQ_DOUBLE(34, ci.f_IG_TIME);
    REQ_DOUBLE(35, ci.f_WINDSPEED);
    REQ_DOUBLE(36, ci.f_DEPTH);
    REQ_DOUBLE(37, ci.f_AMBIENT_TEMP);
    REQ_DOUBLE(38, ci.f_CriInt);
#undef REQ_DOUBLE

    std::string g_fla, g_smo, g_duf;
    if (!parse_group_id(f[39], &g_fla, &row_err) ||
        !parse_group_id(f[40], &g_smo, &row_err) ||
        !parse_group_id(f[41], &g_duf, &row_err)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1
                << ": ef_*_group: " << row_err << "\n";
      return 1;
    }
    // §2a: reset per row, then repopulate all 21 factor fields, always
    // (reproducibility requirement — legacy rows still load and carry them).
    if (!load_factor_block(g_fla, &ci.f_fCO, &ci.f_fCO2, &ci.f_fCH4,
                            &ci.f_fPM25, &ci.f_fPM10, &ci.f_fNOX, &ci.f_fSO2,
                            &row_err) ||
        !load_factor_block(g_smo, &ci.f_sCO, &ci.f_sCO2, &ci.f_sCH4,
                            &ci.f_sPM25, &ci.f_sPM10, &ci.f_sNOX, &ci.f_sSO2,
                            &row_err) ||
        !load_factor_block(g_duf, &ci.f_dCO, &ci.f_dCO2, &ci.f_dCH4,
                            &ci.f_dPM25, &ci.f_dPM10, &ci.f_dNOX, &ci.f_dSO2,
                            &row_err)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1 << ": " << row_err
                << "\n";
      return 1;
    }

    std::string batch = trim(f[42]);
    if (!safe_copy(ci.cr_BatchEqu, batch)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1
                << " field 'batch_equ': value exceeds destination buffer "
                   "capacity (" << (sizeof(ci.cr_BatchEqu) - 1)
                << " chars) — not truncated\n";
      return 1;
    }
    int eq_int;
#define REQ_EQOVR(idx, target)                                             \
  if (!parse_eq_override(f[idx], &eq_int, &row_err)) {                     \
    std::cerr << "[fofem_test] FATAL row " << r + 1 << " field '"          \
              << CONSUME_HEADER[idx] << "': " << row_err << "\n";          \
    return 1;                                                              \
  }                                                                         \
  (target) = eq_int;
    REQ_EQOVR(43, ci.i_EqLit);
    REQ_EQOVR(44, ci.i_EgDufLoa);
    REQ_EQOVR(45, ci.i_EqDufDep);
    REQ_EQOVR(46, ci.i_EqMSE);
    REQ_EQOVR(47, ci.i_EqHerb);
    REQ_EQOVR(48, ci.i_EqShrub);
#undef REQ_EQOVR
#undef REQ_STR

    safe_copy_literal(ci.cr_LoadFN, std::string());
    safe_copy_literal(ci.cr_EmiFN, std::string());

    char cm_err[3000];
    cm_err[0] = '\0';
    int ret = CM_Mngr(&ci, &co, cm_err);
    // Outcome classification for consume: ret==1 -> model succeeded.
    // ret==0 (real error) or ret==2 (no ignition) both count as the row's
    // model error for expect_error purposes (harness-contract §2:
    // "Expected model errors to exercise ... CM_Mngr return 2").
    bool model_errored = (ret != 1);
    std::string err_text = cm_err;
    if (ret == 2 && err_text.empty()) err_text = "Burnup did not ignite";

    Outcome oc = classify(expect_error, model_errored);
    if (oc == Outcome::UNEXPECTED_FAILURE) any_unexpected = true;

    std::vector<std::string> row = {
        f[0], "consume", in.schema_version, outcome_str(oc),
        std::to_string(ret), csv_quote(err_text), in.row_hashes[r]};
    if (oc == Outcome::OK) {
      const float *vals[] = {
          &co.f_LitPre, &co.f_LitCon, &co.f_LitPos,
          &co.f_DW1Pre, &co.f_DW1Con, &co.f_DW1Pos,
          &co.f_DW10Pre, &co.f_DW10Con, &co.f_DW10Pos,
          &co.f_DW100Pre, &co.f_DW100Con, &co.f_DW100Pos,
          &co.f_Snd_DW1kPre, &co.f_Snd_DW1kCon, &co.f_Snd_DW1kPos,
          &co.f_Rot_DW1kPre, &co.f_Rot_DW1kCon, &co.f_Rot_DW1kPos,
          &co.f_DufPre, &co.f_DufCon, &co.f_DufPos, &co.f_DufPer,
          &co.f_HerPre, &co.f_HerCon, &co.f_HerPos,
          &co.f_ShrPre, &co.f_ShrCon, &co.f_ShrPos,
          &co.f_FolPre, &co.f_FolCon, &co.f_FolPos,
          &co.f_BraPre, &co.f_BraCon, &co.f_BraPos,
          &co.f_TotPre, &co.f_TotCon, &co.f_TotPos,
          &co.f_FlaCon, &co.f_SmoCon, &co.f_FlaDur, &co.f_SmoDur,
          &co.f_PM25F, &co.f_PM25S, &co.f_PM10F, &co.f_PM10S,
          &co.f_CH4F, &co.f_CH4S, &co.f_COF, &co.f_COS,
          &co.f_CO2F, &co.f_CO2S, &co.f_NOXF, &co.f_NOXS,
          &co.f_SO2F, &co.f_SO2S,
          &co.f_PM25S_Duff, &co.f_PM10S_Duff, &co.f_CH4S_Duff,
          &co.f_COS_Duff, &co.f_CO2S_Duff, &co.f_NOXS_Duff, &co.f_SO2S_Duff,
          &co.f_MSEPer, &co.f_DufDepPre, &co.f_DufDepCon, &co.f_DufDepPos};
      for (const float *v : vals) row.push_back(fmt(*v));
    } else {
      for (size_t i = 0; i < CONSUME_SCIENTIFIC.size(); ++i)
        row.push_back(NA_SENTINEL);
    }
    if (oc == Outcome::OK && g_output_nonfinite) {
      std::cerr << "[fofem_test] FATAL row " << r + 1
                << ": consume produced a non-finite scientific value on a "
                   "successful row\n";
      return 1;
    }
    summary.row(row);

    if (oc == Outcome::OK) {
      ++ok_rows;
      struct Comp {
        const char *name;
        float pre, con, pos, per;
        int equ;
      };
      Comp comps[] = {
          {"Litter", co.f_LitPre, co.f_LitCon, co.f_LitPos, co.f_LitPer, co.i_LitEqu},
          {"DW1", co.f_DW1Pre, co.f_DW1Con, co.f_DW1Pos, co.f_DW1Per, co.i_DW1Equ},
          {"DW10", co.f_DW10Pre, co.f_DW10Con, co.f_DW10Pos, co.f_DW10Per, co.i_DW10Equ},
          {"DW100", co.f_DW100Pre, co.f_DW100Con, co.f_DW100Pos, co.f_DW100Per, co.i_DW100Equ},
          {"SndDW1k", co.f_Snd_DW1kPre, co.f_Snd_DW1kCon, co.f_Snd_DW1kPos, co.f_Snd_DW1kPer, co.i_Snd_DW1kEqu},
          {"RotDW1k", co.f_Rot_DW1kPre, co.f_Rot_DW1kCon, co.f_Rot_DW1kPos, co.f_Rot_DW1kPer, co.i_Rot_DW1kEqu},
          {"Duff", co.f_DufPre, co.f_DufCon, co.f_DufPos, co.f_DufPer, co.i_DufEqu},
          {"Herb", co.f_HerPre, co.f_HerCon, co.f_HerPos, co.f_HerPer, co.i_HerEqu},
          {"Shrub", co.f_ShrPre, co.f_ShrCon, co.f_ShrPos, co.f_ShrPer, co.i_ShrEqu},
          {"Foliage", co.f_FolPre, co.f_FolCon, co.f_FolPos, co.f_FolPer, co.i_FolEqu},
          {"Branch", co.f_BraPre, co.f_BraCon, co.f_BraPos, co.f_BraPer, co.i_BraEqu},
      };
      for (const auto &c : comps) {
        components.row({f[0], c.name, fmt(c.pre), fmt(c.con), fmt(c.pos),
                         fmt(c.per, 4), std::to_string(c.equ),
                         in.row_hashes[r]});
        ++components_rows_written;
      }
    }
  }

  // Final reconciliation (harness-contract §2b / self-test 11b): the
  // constant-fan-out invariant is exactly 11 component rows per ok row,
  // never more, never fewer.
  if (components_rows_written != 11 * ok_rows) {
    std::cerr << "[fofem_test] FATAL: consume component-row reconciliation "
                 "failed: wrote " << components_rows_written
              << " component rows for " << ok_rows
              << " ok rows (expected " << (11 * ok_rows) << ")\n";
    return 1;
  }
  if (!summary.close_and_check() || !components.close_and_check()) {
    std::cerr << "[fofem_test] FATAL: consume output write/flush/close failed\n";
    return 1;
  }
  return any_unexpected ? 1 : 0;
}

// ===========================================================================
// Mode: litter_eq  (gate0/05-harness-contract.md §3)
// ===========================================================================

static const std::vector<std::string> LITTER_EQ_HEADER = {
    "case_id", "expect_error", "equ", "load_tac", "dw10_moist_pct"};

static int run_litter_eq(const InputFile &in, const std::string &prefix) {
  CsvWriter out(prefix + ".csv");
  if (!out.ok) {
    std::cerr << "[fofem_test] FATAL: cannot open output file\n";
    return 1;
  }
  out.header({"case_id", "mode", "schema_version", "outcome", "con_tac",
              "equ_num", "ret", "err_text", "input_sha256"});
  bool any_unexpected = false;

  for (size_t r = 0; r < in.rows.size(); ++r) {
    const auto &f = in.rows[r];
    g_output_nonfinite = false;  // reset per row; fmt() sets this if called
    std::string row_err;
    bool expect_error = false;
    if (!parse_expect_error(f[1], &expect_error, &row_err)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1 << ": " << row_err << "\n";
      return 1;
    }
    std::string equ = trim(f[2]);
    double load_d;
    if (!parse_strict_double(f[3], &load_d, &row_err)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1 << " field 'load_tac': "
                << row_err << "\n";
      return 1;
    }
    float load = (float)load_d;

    bool model_errored = false;
    float con_tac = 0.0f;
    int equ_num = 0;
    int ret = 1;
    std::string err_text;

    if (equ == "997") {
      double moist_d;
      if (f[4] == NA_SENTINEL) {
        std::cerr << "[fofem_test] FATAL row " << r + 1
                  << ": equ=997 requires dw10_moist_pct, got NA sentinel\n";
        return 1;
      }
      if (!parse_strict_double(f[4], &moist_d, &row_err)) {
        std::cerr << "[fofem_test] FATAL row " << r + 1
                  << " field 'dw10_moist_pct': " << row_err << "\n";
        return 1;
      }
      con_tac = PFW_Litter_Eq997(load, (float)moist_d, &equ_num);
    } else if (equ == "998") {
      if (f[4] != NA_SENTINEL) {
        std::cerr << "[fofem_test] FATAL row " << r + 1
                  << ": equ=998 takes no moisture; dw10_moist_pct must be the "
                  << NA_SENTINEL << " sentinel\n";
        return 1;
      }
      con_tac = LitterSouthEast(load, &equ_num);
    } else {
      // Harness-level dispatch error: no C++ function is bound to this
      // token. Not a fabricated C++ behaviour — there is genuinely nothing
      // to call.
      model_errored = true;
      ret = -1;
      err_text = "unknown equ value '" + equ + "' (expected 997 or 998)";
    }

    Outcome oc = classify(expect_error, model_errored);
    if (oc == Outcome::UNEXPECTED_FAILURE) any_unexpected = true;

    std::string con_tac_fmt = oc == Outcome::OK ? fmt(con_tac) : NA_SENTINEL;
    if (oc == Outcome::OK && g_output_nonfinite) {
      std::cerr << "[fofem_test] FATAL row " << r + 1
                << ": litter_eq produced a non-finite value on a "
                   "successful row\n";
      return 1;
    }
    out.row({f[0], "litter_eq", in.schema_version, outcome_str(oc),
             con_tac_fmt,
             oc == Outcome::OK ? std::to_string(equ_num) : NA_SENTINEL,
             std::to_string(ret), csv_quote(err_text), in.row_hashes[r]});
  }
  if (!out.close_and_check()) {
    std::cerr << "[fofem_test] FATAL: litter_eq output write/flush/close failed\n";
    return 1;
  }
  return any_unexpected ? 1 : 0;
}

// ===========================================================================
// Mode: shrub_herb_eq  (gate0/05-harness-contract.md §4)
// ===========================================================================

static const std::vector<std::string> SHRUB_HERB_EQ_HEADER = {
    "case_id", "expect_error", "region", "cover_group", "season", "fuel_cat",
    "shrub_tac", "herb_tac", "litter_tac", "duff_tac", "duff_moist_pct",
    "crown_fol_tac", "crown_bra_tac", "pct_crown_burn", "force_shrub_equ"};

static int run_shrub_herb_eq(const InputFile &in, const std::string &prefix) {
  CsvWriter out(prefix + ".csv");
  if (!out.ok) {
    std::cerr << "[fofem_test] FATAL: cannot open output file\n";
    return 1;
  }
  out.header({"case_id", "mode", "schema_version", "outcome",
              "shrub_con_tac", "shrub_post_tac", "shrub_pct", "shrub_equ",
              "herb_con_tac", "herb_post_tac", "herb_pct", "herb_equ",
              "fol_con_tac", "fol_post_tac", "fol_pct", "fol_equ",
              "bra_con_tac", "bra_post_tac", "bra_pct", "bra_equ",
              "ret", "err_text", "input_sha256"});
  bool any_unexpected = false;

  for (size_t r = 0; r < in.rows.size(); ++r) {
    const auto &f = in.rows[r];
    g_output_nonfinite = false;  // reset per row; fmt() sets this if called
    std::string row_err;
    bool expect_error = false;
    if (!parse_expect_error(f[1], &expect_error, &row_err)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1 << ": " << row_err << "\n";
      return 1;
    }
    d_CI ci;
    CI_Init(&ci);
#define REQ_STR(idx, target)                                               \
  if (!safe_copy((target), f[idx])) {                                      \
    std::cerr << "[fofem_test] FATAL row " << r + 1 << " field '"          \
              << SHRUB_HERB_EQ_HEADER[idx] << "': value exceeds "          \
                 "destination buffer capacity (" << (sizeof(target) - 1)   \
              << " chars) — not truncated\n";                              \
    return 1;                                                              \
  }
    REQ_STR(2, ci.cr_Region);
    REQ_STR(3, ci.cr_CoverGroup);
    REQ_STR(4, ci.cr_Season);
    REQ_STR(5, ci.cr_FuelCategory);
#undef REQ_STR

    double dv;
#define REQ_DOUBLE(idx, target)                                            \
  if (!parse_strict_double(f[idx], &dv, &row_err)) {                       \
    std::cerr << "[fofem_test] FATAL row " << r + 1 << " field '"          \
              << SHRUB_HERB_EQ_HEADER[idx] << "': " << row_err << "\n";    \
    return 1;                                                              \
  }                                                                         \
  (target) = (float)dv;
    REQ_DOUBLE(6, ci.f_Shrub);
    REQ_DOUBLE(7, ci.f_Herb);
    REQ_DOUBLE(8, ci.f_Lit);
    REQ_DOUBLE(9, ci.f_Duff);
    REQ_DOUBLE(10, ci.f_MoistDuff);
    REQ_DOUBLE(11, ci.f_CroFol);
    REQ_DOUBLE(12, ci.f_CroBra);
    REQ_DOUBLE(13, ci.f_Pc_CroBrn);
#undef REQ_DOUBLE

    int force_shrub_equ = -1;
    if (!parse_eq_override(f[14], &force_shrub_equ, &row_err)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1
                << " field 'force_shrub_equ': " << row_err << "\n";
      return 1;
    }

    float shrub_con = 0, shrub_post = 0, shrub_pct = 0;
    int shrub_equ;
    if (force_shrub_equ >= 0) {
      shrub_con = Shrub_Equ(&ci, force_shrub_equ);
      shrub_equ = force_shrub_equ;
      if (ci.f_Shrub != 0) {
        shrub_post = ci.f_Shrub - shrub_con;
        shrub_pct = (shrub_con / ci.f_Shrub) * 100.0f;
      }
    } else {
      shrub_equ = Calc_Shrub(&ci, &shrub_con, &shrub_post, &shrub_pct);
    }
    bool model_errored = (shrub_con < 0.0f);  // Shrub_Equ's -1 error sentinel

    float herb_con = 0, herb_post = 0, herb_pct = 0;
    int herb_equ = Calc_Herb(&ci, &herb_con, &herb_post, &herb_pct);

    float fol_con = 0, fol_post = 0, fol_pct = 0;
    int fol_equ = Calc_CrownFoliage(&ci, &fol_con, &fol_post, &fol_pct);

    float bra_con = 0, bra_post = 0, bra_pct = 0;
    int bra_equ = Calc_CrownBranch(&ci, &bra_con, &bra_post, &bra_pct);

    std::string err_text;
    int ret = 1;
    if (model_errored) {
      ret = -1;
      err_text = "Shrub_Equ: shrub equation " +
                  std::to_string(force_shrub_equ) + " not implemented";
    }

    Outcome oc = classify(expect_error, model_errored);
    if (oc == Outcome::UNEXPECTED_FAILURE) any_unexpected = true;

    if (oc == Outcome::OK) {
      std::string shrub_con_s = fmt(shrub_con), shrub_post_s = fmt(shrub_post),
                  shrub_pct_s = fmt(shrub_pct, 4);
      std::string herb_con_s = fmt(herb_con), herb_post_s = fmt(herb_post),
                  herb_pct_s = fmt(herb_pct, 4);
      std::string fol_con_s = fmt(fol_con), fol_post_s = fmt(fol_post),
                  fol_pct_s = fmt(fol_pct, 4);
      std::string bra_con_s = fmt(bra_con), bra_post_s = fmt(bra_post),
                  bra_pct_s = fmt(bra_pct, 4);
      if (g_output_nonfinite) {
        std::cerr << "[fofem_test] FATAL row " << r + 1
                  << ": shrub_herb_eq produced a non-finite value on a "
                     "successful row\n";
        return 1;
      }
      out.row({f[0], "shrub_herb_eq", in.schema_version, outcome_str(oc),
               shrub_con_s, shrub_post_s, shrub_pct_s,
               std::to_string(shrub_equ),
               herb_con_s, herb_post_s, herb_pct_s,
               std::to_string(herb_equ),
               fol_con_s, fol_post_s, fol_pct_s,
               std::to_string(fol_equ),
               bra_con_s, bra_post_s, bra_pct_s,
               std::to_string(bra_equ),
               std::to_string(ret), csv_quote(err_text), in.row_hashes[r]});
    } else {
      out.row({f[0], "shrub_herb_eq", in.schema_version, outcome_str(oc),
               NA_SENTINEL, NA_SENTINEL, NA_SENTINEL, NA_SENTINEL,
               NA_SENTINEL, NA_SENTINEL, NA_SENTINEL, NA_SENTINEL,
               NA_SENTINEL, NA_SENTINEL, NA_SENTINEL, NA_SENTINEL,
               NA_SENTINEL, NA_SENTINEL, NA_SENTINEL, NA_SENTINEL,
               std::to_string(ret), csv_quote(err_text), in.row_hashes[r]});
    }
  }
  if (!out.close_and_check()) {
    std::cerr << "[fofem_test] FATAL: shrub_herb_eq output write/flush/close failed\n";
    return 1;
  }
  return any_unexpected ? 1 : 0;
}

// ===========================================================================
// Species table loading (mortality / bark_thick / canopy_cover)
//
// Gate 0 contract correction (Phase 2, direct C++ evidence, Codex-approved).
// The originally approved contract specified MRT_InitST() here. Direct
// evidence contradicts that as the parity/golden initializer:
//
//   - fof_spp.h:10-19 (the sr_MSMT[] table MRT_InitST() copies from):
//     "FOFEM Does Not use this table because the species codes are the old
//      FOFEM5 six letter codes... FuelCalc uses this table..."
//   - sr_MSMT uses codes like "PSEMEN"; the tracked, real production table
//     (src/pyfofem/supporting_data/FOFEM6.7/FOF_SPP.CSV) uses current codes
//     like "PSME", matching pyfofem's own species convention.
//   - Real production callers load species via MRT_LoadSpe(path, "", err)
//     (FOF_DLL/ANSI_MAI.CPP:210-214; FOF_GUI/Wnd_Mort.cpp:481).
//   - MRT_LoadSpe (fof_mrt.h:114/122/130) is a trivial forwarding wrapper
//     around MRT_LoadSpeCSV (fof_mrt.cpp:1139-1145: "i = MRT_LoadSpeCSV
//     (cr_Pth, cr_ErrMes); return i;").
//
// MRT_InitST() remains the sanctioned upstream *empty-path* fallback
// inside MRT_LoadSpeCSV itself (fof_mrt.cpp:993-995: `if (!strcmp(cr_Pth,
// "")) { MRT_InitST(); return 1; }`), but every accepted parity/golden run
// requires an explicit, real species CSV — no empty-path fallback here.
// Gate 0 correctly identified the need to initialize sr_SMT; Phase 2
// direct evidence corrected which initializer is oracle-faithful.
// ===========================================================================

static std::string g_species_table_path;

// Loads the species table via the real production entry point and proves
// (without reimplementing any equation) that it behaves as required before
// any mortality/bark_thick/canopy_cover row is processed.
static bool load_species_table(const std::string &path, std::string *err) {
  if (trim(path).empty()) {
    *err = "species CSV path must not be blank";
    return false;
  }
  {
    std::ifstream probe(path, std::ios::binary);
    if (!probe) {
      *err = "species CSV is not a readable file: " + path;
      return false;
    }
  }
  bool hash_ok = false;
  std::string table_sha256 = sha256_hex_file(path, &hash_ok);
  if (!hash_ok) {
    *err = "could not hash species CSV for provenance: " + path;
    return false;
  }

  std::vector<char> path_buf(path.begin(), path.end());
  path_buf.push_back('\0');
  std::vector<char> ver_buf(1, '\0');
  char load_err_buf[3000];
  load_err_buf[0] = '\0';
  int ret = MRT_LoadSpe(path_buf.data(), ver_buf.data(), load_err_buf);
  std::string load_err = load_err_buf;
  if (ret == 0 || !load_err.empty()) {
    *err = "MRT_LoadSpe failed for '" + path + "': " +
           (load_err.empty() ? std::string("(no message; ret=0)") : load_err);
    return false;
  }

  // --- Startup qualification (Phase 2 amendment §4). None of these calls
  //     reimplement an equation; they only prove the real loaded table and
  //     real pinned lookups behave as required. ---
  const char *qspe = "PSME";  // a current, tracked FOF_SPP.CSV species code

  // (a) a current 4-character code resolves.
  if (SMT_GetIdx((char *)qspe) < 0) {
    *err = std::string("species qualification failed: '") + qspe +
           "' did not resolve after loading " + path;
    return false;
  }

  // (b) SMT_CalcBarkThick succeeds for a known species/DBH, with no
  // unexpected error text left behind even on the success path.
  {
    std::vector<char> spe_buf(qspe, qspe + strlen(qspe));
    spe_buf.push_back('\0');
    char bt_err[3000];
    bt_err[0] = '\0';
    float bark = SMT_CalcBarkThick(spe_buf.data(), 12.0f, bt_err);
    std::string bt_err_text = bt_err;
    if (bark < 0.0f) {
      *err = std::string("species qualification failed: SMT_CalcBarkThick('") +
             qspe + "', 12) returned -1: " + bt_err_text;
      return false;
    }
    if (!bt_err_text.empty()) {
      *err = std::string("species qualification failed: SMT_CalcBarkThick('") +
             qspe + "', 12) left unexpected error text despite success: " +
             bt_err_text;
      return false;
    }
  }

  // (c) canopy equation lookup succeeds for a known species — checked via
  // SMT_Get's own found/not-found return code (not just its output struct
  // contents), and independently confirmed through the real CCT_Get lookup
  // SMT_CalcCrnCov itself relies on (fof_mrt.cpp:1611-1640), not just a
  // nonzero-area heuristic.
  {
    int idx = SMT_GetIdx((char *)qspe);
    d_SMT smt;
    memset(&smt, 0, sizeof(smt));
    int smt_get_ret = SMT_Get(idx, &smt);
    if (smt_get_ret != 1) {
      *err = std::string("species qualification failed: SMT_Get(") +
             std::to_string(idx) + ") for '" + qspe +
             "' returned " + std::to_string(smt_get_ret) + " (not found)";
      return false;
    }
    if (smt.i_No < 0) {
      *err = std::string("species qualification failed: '") + qspe +
             "' has no valid canopy-cover equation index";
      return false;
    }
    d_CCT cct;
    memset(&cct, 0, sizeof(cct));
    int cct_get_ret = CCT_Get(smt.i_No, &cct);
    if (cct_get_ret != 1) {
      *err = std::string("species qualification failed: CCT_Get(") +
             std::to_string(smt.i_No) + ") for '" + qspe +
             "' returned " + std::to_string(cct_get_ret) +
             " (canopy-cover coefficient row not found)";
      return false;
    }
    std::vector<char> spe_buf(qspe, qspe + strlen(qspe));
    spe_buf.push_back('\0');
    float area = SMT_CalcCrnCov(spe_buf.data(), 12.0f, 60.0f);
    if (!(area > 0.0f)) {
      *err = std::string("species qualification failed: SMT_CalcCrnCov('") +
             qspe + "', 12, 60) returned a non-positive area";
      return false;
    }
  }

  // (d) mortality dispatch accepts a current-code scenario.
  {
    d_MIS mis;
    memset(&mis, 0, sizeof(mis));
    safe_copy_literal(mis.cr_Spe, std::string(qspe));
    safe_copy_literal(mis.cr_EquTyp, std::string(e_CroSco));
    mis.f_DBH = 12.0f;
    mis.f_Hgt = 60.0f;
    mis.f_CR = 50.0f;
    mis.f_FS = 4.0f;
    safe_copy_literal(mis.cr_FS, std::string(e_Flame));
    mis.f_BolCha = 0.0f;
    safe_copy_literal(mis.cr_FirSev, std::string());
    mis.f_CKR = 0.0f;
    mis.f_CrnDam = 0.0f;
    safe_copy_literal(mis.cr_BeeDam, std::string(e_BtlNo));
    d_MO mo;
    MO_Init(&mo);
    char mrt_err[3000];
    mrt_err[0] = '\0';
    float prob = MRT_CalcMngr(&mis, &mo, mrt_err);
    std::string mrt_err_text = mrt_err;
    if (!mrt_err_text.empty()) {
      *err = std::string("species qualification failed: MRT_CalcMngr('") +
             qspe + "', CroSco) left unexpected error text: " + mrt_err_text;
      return false;
    }
    if (!(prob >= 0.0f && prob <= 1.0f) || !std::isfinite(prob)) {
      *err = std::string("species qualification failed: MRT_CalcMngr('") +
             qspe + "', CroSco) returned a non-finite or out-of-[0,1] "
                    "probability (" + std::to_string(prob) + ")";
      return false;
    }
  }

  // (e) an obsolete fallback-only code must not substitute for the
  // requested current code — proves this is really the CSV-loaded table,
  // not a leftover/blended MRT_InitST() table.
  const char *obsolete_only = "PSEMEN";
  if (SMT_GetIdx((char *)obsolete_only) >= 0) {
    *err = std::string("species qualification failed: obsolete code '") +
           obsolete_only +
           "' resolved after loading the real species CSV";
    return false;
  }

  g_species_table_path = path;
  std::cout << "SPECIES_TABLE_PATH=" << path << "\n";
  std::cout << "SPECIES_TABLE_SHA256=" << table_sha256 << "\n";
  std::cout << "SPECIES_LOADER=MRT_LoadSpe\n";
  std::cout << "SPECIES_LOADER_SOURCE=FOF_UNIX/fof_mrt.cpp:1139-1145\n";
  std::cout << "SPECIES_TABLE_ROLE=FOFEM mortality/species equation table\n";
  return true;
}

// ===========================================================================
// Mode: mortality  (gate0/05-harness-contract.md §5)
// ===========================================================================

static const std::vector<std::string> MORTALITY_HEADER = {
    "case_id", "expect_error", "species", "equ_type", "dbh_in", "ht_ft",
    "crown_ratio_x10", "fs_value_ft", "fs_kind", "bole_char_ft",
    "fire_severity", "ckr_pct", "cvk_pct", "beetles"};

static int run_mortality(const InputFile &in, const std::string &prefix) {
  // Species table is loaded and qualified once in main() before dispatch
  // (see load_species_table() above).
  CsvWriter out(prefix + ".csv");
  if (!out.ok) {
    std::cerr << "[fofem_test] FATAL: cannot open output file\n";
    return 1;
  }
  out.header({"case_id", "mode", "schema_version", "outcome", "prob",
              "mort_equ", "ret", "err_text", "input_sha256"});
  bool any_unexpected = false;

  for (size_t r = 0; r < in.rows.size(); ++r) {
    const auto &f = in.rows[r];
    g_output_nonfinite = false;  // reset per row; fmt() sets this if called
    std::string row_err;
    bool expect_error = false;
    if (!parse_expect_error(f[1], &expect_error, &row_err)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1 << ": " << row_err << "\n";
      return 1;
    }

    d_MIS mis;
    memset(&mis, 0, sizeof(mis));
    if (!safe_copy(mis.cr_Spe, f[2])) {
      std::cerr << "[fofem_test] FATAL row " << r + 1
                << " field 'species': value exceeds destination buffer "
                   "capacity (" << (sizeof(mis.cr_Spe) - 1)
                << " chars) — not truncated\n";
      return 1;
    }

    std::string equ_type = trim(f[3]);
    if (equ_type == "CroSco") safe_copy_literal(mis.cr_EquTyp, std::string(e_CroSco));
    else if (equ_type == "CroDam") safe_copy_literal(mis.cr_EquTyp, std::string(e_CroDam));
    else if (equ_type == "BolCha") safe_copy_literal(mis.cr_EquTyp, std::string(e_BolCha));
    else {
      std::cerr << "[fofem_test] FATAL row " << r + 1
                << ": equ_type must be CroSco, CroDam, or BolCha, got '"
                << equ_type << "'\n";
      return 1;
    }

    double dv;
#define REQ_DOUBLE(idx, target)                                            \
  if (!parse_strict_double(f[idx], &dv, &row_err)) {                       \
    std::cerr << "[fofem_test] FATAL row " << r + 1 << " field '"          \
              << MORTALITY_HEADER[idx] << "': " << row_err << "\n";        \
    return 1;                                                              \
  }                                                                         \
  (target) = (float)dv;
    REQ_DOUBLE(4, mis.f_DBH);
    REQ_DOUBLE(5, mis.f_Hgt);
    REQ_DOUBLE(6, mis.f_CR);
    REQ_DOUBLE(7, mis.f_FS);
#undef REQ_DOUBLE

    std::string fs_kind = trim(f[8]);
    if (fs_kind == "Flame") safe_copy_literal(mis.cr_FS, std::string(e_Flame));
    else if (fs_kind == "Scorch") safe_copy_literal(mis.cr_FS, std::string(e_Scorch));
    else {
      std::cerr << "[fofem_test] FATAL row " << r + 1
                << ": fs_kind must be Flame or Scorch, got '" << fs_kind
                << "'\n";
      return 1;
    }

    if (!parse_strict_double(f[9], &dv, &row_err)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1
                << " field 'bole_char_ft': " << row_err << "\n";
      return 1;
    }
    mis.f_BolCha = (float)dv;

    std::string fire_sev = trim(f[10]);
    if (fire_sev == NA_SENTINEL) fire_sev = "";
    if (!safe_copy(mis.cr_FirSev, fire_sev)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1
                << " field 'fire_severity': value exceeds destination "
                   "buffer capacity (" << (sizeof(mis.cr_FirSev) - 1)
                << " chars) — not truncated\n";
      return 1;
    }

    if (!parse_strict_double(f[11], &dv, &row_err)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1
                << " field 'ckr_pct': " << row_err << "\n";
      return 1;
    }
    mis.f_CKR = (float)dv;

    if (!parse_strict_double(f[12], &dv, &row_err)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1
                << " field 'cvk_pct': " << row_err << "\n";
      return 1;
    }
    mis.f_CrnDam = (float)dv;

    std::string beetles_raw = trim(f[13]);
    if (beetles_raw != "0" && beetles_raw != "1") {
      std::cerr << "[fofem_test] FATAL row " << r + 1
                << ": beetles must be exactly 0 or 1\n";
      return 1;
    }
    safe_copy_literal(mis.cr_BeeDam, std::string(beetles_raw == "1" ? e_BtlYes : e_BtlNo));

    d_MO mo;
    MO_Init(&mo);
    char err_buf[3000];
    err_buf[0] = '\0';
    float prob = MRT_CalcMngr(&mis, &mo, err_buf);
    bool model_errored = (prob < 0.0f);
    int ret = model_errored ? -1 : 1;

    Outcome oc = classify(expect_error, model_errored);
    if (oc == Outcome::UNEXPECTED_FAILURE) any_unexpected = true;

    std::string prob_fmt = oc == Outcome::OK ? fmt(prob, 6) : NA_SENTINEL;
    if (oc == Outcome::OK && g_output_nonfinite) {
      std::cerr << "[fofem_test] FATAL row " << r + 1
                << ": mortality produced a non-finite probability on a "
                   "successful row\n";
      return 1;
    }
    out.row({f[0], "mortality", in.schema_version, outcome_str(oc),
             prob_fmt,
             oc == Outcome::OK ? std::string(mo.cr_MortEqu) : NA_SENTINEL,
             std::to_string(ret), csv_quote(err_buf), in.row_hashes[r]});
  }
  if (!out.close_and_check()) {
    std::cerr << "[fofem_test] FATAL: mortality output write/flush/close failed\n";
    return 1;
  }
  return any_unexpected ? 1 : 0;
}

// ===========================================================================
// Mode: bark_thick  (gate0/05-harness-contract.md §5a)
// ===========================================================================

static const std::vector<std::string> BARK_THICK_HEADER = {
    "case_id", "expect_error", "species", "dbh_in"};

static int run_bark_thick(const InputFile &in, const std::string &prefix) {
  // Species table is loaded and qualified once in main() before dispatch
  // (see load_species_table() above).
  CsvWriter out(prefix + ".csv");
  if (!out.ok) {
    std::cerr << "[fofem_test] FATAL: cannot open output file\n";
    return 1;
  }
  out.header({"case_id", "mode", "schema_version", "outcome", "bark_thick_in",
              "ret", "err_text", "input_sha256"});
  bool any_unexpected = false;

  for (size_t r = 0; r < in.rows.size(); ++r) {
    const auto &f = in.rows[r];
    g_output_nonfinite = false;  // reset per row; fmt() sets this if called
    std::string row_err;
    bool expect_error = false;
    if (!parse_expect_error(f[1], &expect_error, &row_err)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1 << ": " << row_err << "\n";
      return 1;
    }
    double dbh_d;
    if (!parse_strict_double(f[3], &dbh_d, &row_err)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1 << " field 'dbh_in': "
                << row_err << "\n";
      return 1;
    }
    std::vector<char> spe(f[2].begin(), f[2].end());
    spe.push_back('\0');
    char err_buf[3000];
    err_buf[0] = '\0';
    float bark = SMT_CalcBarkThick(spe.data(), (float)dbh_d, err_buf);
    bool model_errored = (bark < 0.0f);
    int ret = model_errored ? -1 : 1;

    Outcome oc = classify(expect_error, model_errored);
    if (oc == Outcome::UNEXPECTED_FAILURE) any_unexpected = true;

    std::string bark_fmt = oc == Outcome::OK ? fmt(bark, 6) : NA_SENTINEL;
    if (oc == Outcome::OK && g_output_nonfinite) {
      std::cerr << "[fofem_test] FATAL row " << r + 1
                << ": bark_thick produced a non-finite value on a "
                   "successful row\n";
      return 1;
    }
    out.row({f[0], "bark_thick", in.schema_version, outcome_str(oc),
             bark_fmt,
             std::to_string(ret), csv_quote(err_buf), in.row_hashes[r]});
  }
  if (!out.close_and_check()) {
    std::cerr << "[fofem_test] FATAL: bark_thick output write/flush/close failed\n";
    return 1;
  }
  return any_unexpected ? 1 : 0;
}

// ===========================================================================
// Mode: canopy_cover  (gate0/05-harness-contract.md §6)
// ===========================================================================

static const std::vector<std::string> CANOPY_COVER_HEADER = {
    "case_id", "expect_error", "stand_id", "species", "dbh_in", "ht_ft"};

struct TreeRow {
  std::string case_id, stand_id;
  Outcome outcome;
  float crown_area_ft2 = 0.0f;
  int cct_equ_no = -1;
  int ret = 1;
  std::string err_text;
  std::string input_hash;
};

static int run_canopy_cover(const InputFile &in, const std::string &prefix) {
  // Species table is loaded and qualified once in main() before dispatch
  // (see load_species_table() above).
  CsvWriter trees(prefix + "_trees.csv");
  CsvWriter stands(prefix + "_stands.csv");
  CsvWriter groups(prefix + "_groups.csv");
  if (!trees.ok || !stands.ok || !groups.ok) {
    std::cerr << "[fofem_test] FATAL: cannot open output files\n";
    return 1;
  }
  trees.header({"case_id", "stand_id", "mode", "schema_version", "outcome",
                "crown_area_ft2", "cct_equ_no", "ret", "err_text",
                "input_sha256"});
  stands.header({"stand_id", "mode", "schema_version", "n_trees",
                  "total_area_ft2", "pct_cover", "stand_sha256"});
  groups.header({"stand_id", "mode", "schema_version", "n_members", "n_ok",
                  "n_expected_model_error", "n_unexpected_failure",
                  "aggregate_emitted", "suppression_reason", "group_sha256"});

  bool any_unexpected = false;
  std::vector<TreeRow> processed;

  // Contiguity check (self-test 19): stand_id runs must not be interleaved.
  std::map<std::string, bool> seen_stand_closed;
  std::string last_stand;

  for (size_t r = 0; r < in.rows.size(); ++r) {
    const auto &f = in.rows[r];
    g_output_nonfinite = false;  // reset per row; fmt() sets this if called
    std::string row_err;
    bool expect_error = false;
    if (!parse_expect_error(f[1], &expect_error, &row_err)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1 << ": " << row_err << "\n";
      return 1;
    }
    std::string stand_id = f[2];
    if (!last_stand.empty() && stand_id != last_stand) {
      seen_stand_closed[last_stand] = true;
    }
    if (seen_stand_closed.count(stand_id)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1
                << ": stand_id '" << stand_id
                << "' rows are not contiguous\n";
      return 1;
    }
    last_stand = stand_id;

    double dbh_d, ht_d;
    if (!parse_strict_double(f[4], &dbh_d, &row_err)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1 << " field 'dbh_in': "
                << row_err << "\n";
      return 1;
    }
    if (!parse_strict_double(f[5], &ht_d, &row_err)) {
      std::cerr << "[fofem_test] FATAL row " << r + 1 << " field 'ht_ft': "
                << row_err << "\n";
      return 1;
    }

    TreeRow tr;
    tr.case_id = f[0];
    tr.stand_id = stand_id;
    tr.input_hash = in.row_hashes[r];

    std::vector<char> spe(f[3].begin(), f[3].end());
    spe.push_back('\0');
    int idx = SMT_GetIdx(spe.data());
    bool model_errored = false;
    if (idx < 0) {
      // Harness-side guard: SMT_CalcCrnCov does not itself check iX<0 before
      // indexing sr_SMT[iX] (fof_mrt.cpp:1611-1640) — calling it with an
      // unresolved species would read out of bounds. The harness must not
      // call it in that case; this is a fail-closed harness contract, not a
      // fabricated C++ behaviour.
      model_errored = true;
      tr.ret = -1;
      tr.err_text = "unknown species '" + f[3] + "' (SMT_GetIdx < 0)";
    } else {
      tr.crown_area_ft2 = SMT_CalcCrnCov(spe.data(), (float)dbh_d, (float)ht_d);
      d_SMT smt;
      memset(&smt, 0, sizeof(smt));
      SMT_Get(idx, &smt);
      tr.cct_equ_no = smt.i_No;
      tr.ret = 1;
    }

    tr.outcome = classify(expect_error, model_errored);
    if (tr.outcome == Outcome::UNEXPECTED_FAILURE) any_unexpected = true;
    processed.push_back(tr);

    std::string area_fmt = tr.outcome == Outcome::OK ? fmt(tr.crown_area_ft2, 4) : NA_SENTINEL;
    if (tr.outcome == Outcome::OK && g_output_nonfinite) {
      std::cerr << "[fofem_test] FATAL row " << r + 1
                << ": canopy_cover produced a non-finite crown area on a "
                   "successful row\n";
      return 1;
    }
    trees.row({tr.case_id, tr.stand_id, "canopy_cover", in.schema_version,
               outcome_str(tr.outcome),
               area_fmt,
               tr.outcome == Outcome::OK ? std::to_string(tr.cct_equ_no) : NA_SENTINEL,
               std::to_string(tr.ret), csv_quote(tr.err_text), tr.input_hash});
  }

  // Group by stand_id, preserving first-seen order (rows are contiguous per
  // stand, enforced above).
  std::vector<std::string> stand_order;
  std::map<std::string, std::vector<size_t>> stand_members;
  for (size_t i = 0; i < processed.size(); ++i) {
    const std::string &sid = processed[i].stand_id;
    if (!stand_members.count(sid)) stand_order.push_back(sid);
    stand_members[sid].push_back(i);
  }

  // Self-test-only fault injection (self-test row 11g): FOFEM_TEST_FAULT
  // must never be set in a real run. It lets the self-test suite actually
  // *inject* an inconsistent aggregate/membership state and prove the
  // reconciliation pass below rejects it, rather than merely observing
  // the two states the normal code path can construct on its own
  // (all-ok and mixed) — which cannot demonstrate rejection of a state the
  // normal code path can never produce.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv is fine for a read-only test-only flag
#endif
  const char *fault_env = std::getenv("FOFEM_TEST_FAULT");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  bool inject_aggregate_mismatch =
      fault_env && std::string(fault_env) == "canopy_aggregate_mismatch";

  struct GroupResult {
    std::string sid, group_sha, suppression_reason;
    int n_members = 0, n_ok = 0, n_eme = 0, n_uf = 0;
    bool emit_aggregate = false;
    float total_area = 0.0f, pct_cover = 0.0f;
  };
  std::vector<GroupResult> group_results;

  for (size_t stand_ix = 0; stand_ix < stand_order.size(); ++stand_ix) {
    const std::string &sid = stand_order[stand_ix];
    const auto &member_idx = stand_members[sid];
    GroupResult gr;
    gr.sid = sid;
    std::string concat_hashes;
    for (size_t i : member_idx) {
      const TreeRow &tr = processed[i];
      if (tr.outcome == Outcome::OK) { ++gr.n_ok; gr.total_area += tr.crown_area_ft2; }
      else if (tr.outcome == Outcome::EXPECTED_MODEL_ERROR) ++gr.n_eme;
      else ++gr.n_uf;
      concat_hashes += tr.input_hash;
    }
    gr.group_sha = sha256_hex(concat_hashes);
    gr.n_members = (int)member_idx.size();
    gr.emit_aggregate = (gr.n_ok == gr.n_members);

    if (inject_aggregate_mismatch && stand_ix == 0) {
      // Deliberately violate the invariant for the reconciliation
      // self-test below: force emit_aggregate=true even though not every
      // member is ok (only possible if this group actually has a
      // non-ok member; the self-test that sets this env var supplies
      // input guaranteeing that).
      gr.emit_aggregate = true;
    }

    gr.suppression_reason = "none";
    if (!gr.emit_aggregate) {
      gr.suppression_reason = (gr.n_uf > 0) ? "unexpected_failure_member"
                                             : "expected_model_error_member";
    }
    if (gr.emit_aggregate) {
      gr.pct_cover = 100.0f * (1.0f - expf(-(gr.total_area / 43560.0f)));
    }
    group_results.push_back(gr);
  }

  // Final reconciliation (harness-contract §1, self-test 11g): every
  // group's emit_aggregate MUST equal (n_ok == n_members) exactly, and
  // total per-group membership must reconcile against the processed tree
  // count. Any violation is a hard failure — no output is trusted.
  int total_members_reconciled = 0;
  for (const GroupResult &gr : group_results) {
    bool invariant_holds = (gr.emit_aggregate == (gr.n_ok == gr.n_members));
    if (!invariant_holds) {
      std::cerr << "[fofem_test] FATAL: canopy_cover group '" << gr.sid
                << "' failed aggregate-reconciliation: emit_aggregate="
                << gr.emit_aggregate << " but n_ok=" << gr.n_ok
                << " n_members=" << gr.n_members << "\n";
      return 1;
    }
    if (gr.n_ok + gr.n_eme + gr.n_uf != gr.n_members) {
      std::cerr << "[fofem_test] FATAL: canopy_cover group '" << gr.sid
                << "' member-count reconciliation failed: "
                << gr.n_ok << "+" << gr.n_eme << "+" << gr.n_uf
                << " != " << gr.n_members << "\n";
      return 1;
    }
    total_members_reconciled += gr.n_members;
  }
  if (total_members_reconciled != (int)processed.size()) {
    std::cerr << "[fofem_test] FATAL: canopy_cover total group membership ("
              << total_members_reconciled << ") does not reconcile against "
              << "total processed tree rows (" << processed.size() << ")\n";
    return 1;
  }

  for (const GroupResult &gr : group_results) {
    if (gr.emit_aggregate) {
      stands.row({gr.sid, "canopy_cover", in.schema_version,
                  std::to_string(gr.n_members), fmt(gr.total_area, 4),
                  fmt(gr.pct_cover, 4), gr.group_sha});
      if (g_output_nonfinite) {
        std::cerr << "[fofem_test] FATAL: canopy_cover group '" << gr.sid
                  << "' produced a non-finite aggregate value\n";
        return 1;
      }
    }
    groups.row({gr.sid, "canopy_cover", in.schema_version,
                std::to_string(gr.n_members), std::to_string(gr.n_ok),
                std::to_string(gr.n_eme), std::to_string(gr.n_uf),
                gr.emit_aggregate ? "1" : "0", gr.suppression_reason, gr.group_sha});
  }

  if (!trees.close_and_check() || !stands.close_and_check() || !groups.close_and_check()) {
    std::cerr << "[fofem_test] FATAL: canopy_cover output write/flush/close failed\n";
    return 1;
  }

  return any_unexpected ? 1 : 0;
}

// ===========================================================================
// Mode dispatch table
// ===========================================================================

struct ModeSpec {
  const char *name;
  const std::vector<std::string> *header;
  int (*run)(const InputFile &, const std::string &);
};

static const ModeSpec MODES[] = {
    {"consume", &CONSUME_HEADER, run_consume},
    {"litter_eq", &LITTER_EQ_HEADER, run_litter_eq},
    {"shrub_herb_eq", &SHRUB_HERB_EQ_HEADER, run_shrub_herb_eq},
    {"mortality", &MORTALITY_HEADER, run_mortality},
    {"bark_thick", &BARK_THICK_HEADER, run_bark_thick},
    {"canopy_cover", &CANOPY_COVER_HEADER, run_canopy_cover},
};

int main(int argc, char **argv) {
  // Hidden self-test mode (audit finding #5): print SHA-256 known vectors
  // for independent cross-checking against a trusted implementation
  // (e.g. Python hashlib), and optionally a file digest. Not part of the
  // documented six-mode harness contract.
  if (argc >= 2 && std::string(argv[1]) == "--selftest-sha256") {
    std::cout << "SHA256_EMPTY=" << sha256_hex("") << "\n";
    std::cout << "SHA256_ABC=" << sha256_hex("abc") << "\n";
    if (argc >= 3) {
      bool ok = false;
      std::string h = sha256_hex_file(argv[2], &ok);
      if (!ok) {
        std::cerr << "[fofem_test] FATAL: cannot hash file: " << argv[2]
                  << "\n";
        return 1;
      }
      std::cout << "SHA256_FILE=" << h << "\n";
    }
    return 0;
  }

  if (argc < 3) {
    std::cerr << "Usage: fofem_test <input.csv> <output_prefix> "
                 "[--species-csv <path>]\n";
    return 1;
  }
  std::string input_path = argv[1];
  std::string output_prefix = argv[2];

  // Parse the optional --species-csv <path> flag. Applicability (required
  // for mortality/bark_thick/canopy_cover, rejected for every other mode)
  // is checked below once the mode is known.
  bool have_species_csv = false;
  std::string species_csv_path;
  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--species-csv") {
      if (have_species_csv) {
        std::cerr << "[fofem_test] FATAL: --species-csv given more than once\n";
        return 1;
      }
      if (i + 1 >= argc) {
        std::cerr << "[fofem_test] FATAL: --species-csv requires a path "
                     "argument\n";
        return 1;
      }
      species_csv_path = argv[++i];
      have_species_csv = true;
    } else {
      std::cerr << "[fofem_test] FATAL: unknown option '" << arg << "'\n";
      return 1;
    }
  }

  // Peek the magic line to find the declared mode before we know which
  // header to validate against.
  std::ifstream peek(input_path);
  if (!peek) {
    std::cerr << "[fofem_test] FATAL: cannot open input file: " << input_path
              << "\n";
    return 1;
  }
  std::string magic_line;
  if (!std::getline(peek, magic_line)) {
    std::cerr << "[fofem_test] FATAL: empty input file (no magic/version line)\n";
    return 1;
  }
  peek.close();
  if (!magic_line.empty() && magic_line.back() == '\r') magic_line.pop_back();
  std::vector<std::string> magic_fields = split_comma(magic_line);
  if (magic_fields.size() != 3 || magic_fields[0] != "#fofem-harness") {
    std::cerr << "[fofem_test] FATAL: malformed magic/version line: "
              << magic_line << "\n";
    return 1;
  }
  std::string mode = magic_fields[1];
  if (magic_fields[2] != "1") {
    std::cerr << "[fofem_test] FATAL: unsupported schema_version '"
              << magic_fields[2] << "' (only '1' is defined)\n";
    return 1;
  }

  const ModeSpec *spec = nullptr;
  for (const auto &m : MODES) {
    if (mode == m.name) { spec = &m; break; }
  }
  if (!spec) {
    std::cerr << "[fofem_test] FATAL: unknown mode '" << mode << "'\n";
    return 1;
  }

  bool mode_needs_species =
      (mode == "mortality" || mode == "bark_thick" || mode == "canopy_cover");
  if (mode_needs_species && !have_species_csv) {
    std::cerr << "[fofem_test] FATAL: mode '" << mode
              << "' requires --species-csv <path>\n";
    return 1;
  }
  if (!mode_needs_species && have_species_csv) {
    std::cerr << "[fofem_test] FATAL: mode '" << mode
              << "' does not accept --species-csv\n";
    return 1;
  }
  if (mode_needs_species) {
    std::string species_err;
    if (!load_species_table(species_csv_path, &species_err)) {
      std::cerr << "[fofem_test] FATAL: " << species_err << "\n";
      return 1;
    }
  }

  InputFile infile;
  std::string err;
  if (!read_input_file(input_path, *spec->header, &infile, &err)) {
    std::cerr << "[fofem_test] FATAL: " << err << "\n";
    return 1;
  }

  return spec->run(infile, output_prefix);
}
