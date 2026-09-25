/* massman_fof_dll_probe_driver.cpp -- tracked diagnostic reproduction
 * artifact for findings F-55/F-57/F-58 (gate0/04-findings.md).
 *
 * This is NOT part of the pinned reference/fofem_cpp submodule and NOT an
 * overlay change to any tracked upstream file. It is a standalone driver,
 * compiled/linked by massman_fof_dll_probe.py directly against the pinned
 * FOF_DLL/*.cpp sources (by absolute path, from a scratch build directory
 * rooted under this repository's own tree) to give independent,
 * reproducible evidence for the build/link/run-feasibility half of F-55.
 *
 * It calls ONLY the documented public lifecycle FOF_DLL/BMSoil.h and
 * FOF_DLL/HTAA.h declare:
 *   BMI_Init(&bmi) -> populate documented d_BMI fields -> HTA_Init() ->
 *   HMV_Model(&bmi, "") -> loop every (layer, time-index) pair via
 *   HTA_Count()/HTA_Layers()/HTA_Get().
 * No equation is copied or reimplemented here.
 *
 * Measurement scope (per the Phase 6 probe-hardening pass): every saved
 * (layer, time-index) sample across EVERY layer HTA_Layers() reports, not
 * only layer 1's first/last sample -- so the finiteness claim this probe
 * supports is exactly "every saved sample this driver inspected", stated
 * precisely, never generalised beyond what was actually measured.
 *
 * Output is a single line of '|'-separated machine-readable fields (never
 * prose) so the Python wrapper can parse it without depending on any
 * particular libc's float-formatting of NaN/Inf:
 *   PROBE_RESULT|hmv_rc=<int>|errmes=<escaped>|hta_count=<int>|
 *   hta_layers=<int>|hta_get_failed=<0|1>|
 *   heat_total=<int>|heat_finite=<int>|heat_any_nan=<0|1>|heat_any_inf=<0|1>|
 *   heat_first=<repr>|heat_last=<repr>|
 *   moist_total=...|moist_finite=...|moist_any_nan=...|moist_any_inf=...|
 *   moist_first=...|moist_last=...|
 *   psin_total=...|psin_finite=...|psin_any_nan=...|psin_any_inf=...|
 *   psin_first=...|psin_last=...|
 *   time_total=...|time_finite=...|time_any_nan=...|time_any_inf=...|
 *   time_first=...|time_last=...
 */
#include <cmath>
#include <cstdio>
#include <cstring>

#include "BMSoil.h"
#include "HTAA.h"

namespace {

/* Per-field running accounting across every (layer, time-index) sample
 * this driver inspects. */
struct FieldStats {
    long total = 0;
    long finite = 0;
    int any_nan = 0;
    int any_inf = 0;
    float first = 0.0f;
    float last = 0.0f;
    bool have_first = false;
};

void observe(FieldStats &s, float value)
{
    s.total += 1;
    if (std::isfinite(value)) {
        s.finite += 1;
    } else if (std::isnan(value)) {
        s.any_nan = 1;
    } else {
        s.any_inf = 1;
    }
    if (!s.have_first) {
        s.first = value;
        s.have_first = true;
    }
    s.last = value;
}

/* Escape '|' and control characters out of cr_ErrMes so the single-line
 * machine-readable result cannot be corrupted by unexpected C++ error
 * text. */
void print_escaped(const char *s)
{
    for (const char *p = s; *p; ++p) {
        unsigned char c = static_cast<unsigned char>(*p);
        if (c == '|' || c == '\n' || c == '\r' || c < 0x20) {
            std::printf("\\x%02x", c);
        } else {
            std::putchar(c);
        }
    }
}

/* MSVC's CRT can format a negative-signed NaN bit pattern as
 * "-nan(ind)" ("indefinite"), which Python's float() cannot parse, and
 * that exact bit pattern is exactly what this driver's own uninitialized-
 * float defaults / degenerate-solver arithmetic produces. NaN's sign bit
 * carries no meaning, so canonicalise to a single "nan" spelling
 * (matching the plain "nan"/"inf"/"-inf" spellings Python's own float()
 * parser accepts) rather than depend on any libc's raw %g rendering. */
void print_float(double value)
{
    if (std::isnan(value)) {
        std::printf("nan");
    } else if (std::isinf(value)) {
        std::printf(value > 0 ? "inf" : "-inf");
    } else {
        std::printf("%.6g", value);
    }
}

void print_field(const char *name, const FieldStats &s)
{
    std::printf(
        "|%s_total=%ld|%s_finite=%ld|%s_any_nan=%d|%s_any_inf=%d|",
        name, s.total, name, s.finite, name, s.any_nan, name, s.any_inf);
    std::printf("%s_first=", name);
    print_float(static_cast<double>(s.first));
    std::printf("|%s_last=", name);
    print_float(static_cast<double>(s.last));
}

} // namespace

int main()
{
    d_BMI bmi;
    BMI_Init(&bmi);

    /* Documented-bounds-compliant inputs (BMSoil.h e_Min.../e_Max... macros):
     * moisture 0.10 in [0.01, 0.25]; bulk density 1.25 in [0.70, 1.8];
     * particle density 2.65 in [2.3, 2.9]; burn time 1.0 hr in
     * [0.25, 100]; time-to-peak 0.5 hr in [0.05, 4.0]. */
    bmi.f_Moist = 0.10f;
    bmi.f_SoiBulDen = 1.25f;
    bmi.f_SoiParDen = 2.65f;
    bmi.f_AmbAirTmp = 20.0f;
    bmi.f_Qabs = 31.0f;
    bmi.d_SimTime = 60.0;
    bmi.f_BurnTime = 1.0f;
    bmi.f_MaxWatTim = 0.5f;
    std::strcpy(bmi.cr_TemMoi, "Temp");
    std::strcpy(bmi.cr_FirTyp, e_FT_Test);

    HTA_Init();

    int rc = HMV_Model(&bmi, "");

    int count = HTA_Count();
    int layers = HTA_Layers();
    int hta_get_failed = 0;

    FieldStats heat, moist, psin, time_s;

    for (int lay = 1; lay <= layers && !hta_get_failed; ++lay) {
        for (int ix = 0; ix < count; ++ix) {
            float h = 0.0f, m = 0.0f, p = 0.0f, t = 0.0f;
            int ok = HTA_Get(lay, ix, &h, &m, &p, &t);
            if (!ok) {
                hta_get_failed = 1;
                break;
            }
            observe(heat, h);
            observe(moist, m);
            observe(psin, p);
            observe(time_s, t);
        }
    }

    std::printf("PROBE_RESULT|hmv_rc=%d|errmes=", rc);
    print_escaped(bmi.cr_ErrMes);
    std::printf("|hta_count=%d|hta_layers=%d|hta_get_failed=%d",
                count, layers, hta_get_failed);
    print_field("heat", heat);
    print_field("moist", moist);
    print_field("psin", psin);
    print_field("time", time_s);
    std::printf("\n");

    return 0;
}
