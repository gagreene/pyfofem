/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: fof_soi.c
* Desc: this was Soilheat.pas
*       This is used by Duff Sim & Exp Heat
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include  "fof_sgv.h"
#include  "fof_sh.h"
#include  "fof_sd.h"
#include  "fof_soi.h"

extern  char gcr_SoiErr[];

/*===========================================================================
 * F-70 diagnostic-observer instrumentation (2026-09-18 pass).
 *
 * THIS FILE IS A DIAGNOSTICS-ONLY INSTRUMENTED OBSERVER, NOT A DISTINCT
 * ORACLE. It is a byte-for-byte copy of the pinned
 * reference/fofem_cpp/FOF_UNIX/fof_soi.cpp (verified identical via
 * sha256sum before this comment block was added: both files hashed
 * f70106c5d20e3a4aeedf412e35019456bd3bc003d1df87a1d4069b4d51874f30),
 * with exactly THREE additions, all pure hook calls plus their own
 * diagnostic-only local snapshots -- no equation, no arithmetic, and no
 * existing variable's VALUE is changed anywhere. The only control-flow
 * change: two originally single-statement `if (rr_p[i] > 0) stmt;` /
 * `if (rr_p[i] < -1e20) stmt;` bodies (fof_soi.cpp:150-153) are wrapped
 * in `{ }` so a diagnostic-only clamp-branch flag can sit next to the
 * UNCHANGED statement -- `diff` confirms each wrapped statement is
 * byte-identical to the pinned source, only the enclosing braces and an
 * adjacent new line are added:
 *   1. SoiDiagRecordTimestep(): inserted at the end of soiltemp_step()
 *      (see its own "F-70 DIAGNOSTIC HOOK" comment below), right after
 *      the real "commit" step (fof_soi.cpp:195-206) and immediately
 *      before the function's own `return true;` -- fires once per
 *      converged TIMESTEP.
 *   2. SoiDiagRecordSubIteration(): inserted right after `iN_SoilBug++`
 *      (fof_soi.cpp:177 area, "Change 11-6-05" block) and BEFORE the
 *      convergence break-check -- fires once per Newton SUB-ITERATION
 *      (added in a second same-pass round, once the end-of-step-only
 *      hook proved insufficient to localise a large first-timestep
 *      divergence found in the non-duff route -- see F-70's write-up).
 *   3. SoiDiagRecordSurfaceUpdate(): inserted inside the main per-node
 *      for-loop, gated `if (i == 1)`, immediately after the surface
 *      node's own `rr_h[1] = humidity(...)` call (fof_soi.cpp:191 area)
 *      -- fires once per Newton SUB-ITERATION and captures EVERY named
 *      intermediate quantity the surface-node (i=1) update itself reads
 *      or writes this sub-iteration (old/new temperature and matric
 *      potential, water content, humidity, vapor pressure, all
 *      conductivity terms, the boundary Stefan-Boltzmann correction
 *      before/after, the residual numerators, the Jacobian/denominator
 *      terms, both Newton increments, and any clamp branch taken) --
 *      added in a THIRD same-pass round because
 *      SoiDiagRecordSubIteration's 5 fields (tn1/p1/r_sev/r_seh only)
 *      are insufficient to isolate WHICH named quantity first diverges
 *      within one sub-iteration, only that one does.
 * This file lives ONLY under reference/fofem_cpp_overlay/ (an overlay
 * source, maintained by pyfofem, never inside reference/fofem_cpp/ the
 * pinned submodule) and is compiled into a SEPARATE CMake target
 * (fofem_test_soidiag) alongside every other pinned FOF_UNIX/*.cpp file
 * UNCHANGED -- the normal fofem_test target continues to compile the
 * REAL, unmodified reference/fofem_cpp/FOF_UNIX/fof_soi.cpp and is
 * therefore byte-identical in behavior to before this pass (see the
 * "identical" proof in gate0/04-findings.md F-70 / docs/CODEBASE.md).
 *
 * All three hook functions are declared here but DEFINED in
 * test_harness.cpp (shared, unmodified, between both targets) -- the
 * normal fofem_test target links those same definitions but the
 * symbols are simply never called there (its own fof_soi.cpp has no
 * such calls), so their presence has zero effect on normal output.
 *===========================================================================*/
extern "C" {
void SoiDiagRecordTimestep(
    int node_count,
    const float *tn, const float *t_true,
    const float *wn, const float *w_true,
    const float *p, const float *h, const float *psat,
    const float *kh, const float *kv, const float *enh,
    float r_sev, float r_seh, int n_subiter, float r_rabs_in);
void SoiDiagRecordSubIteration(
    int n_subiter, float tn1, float p1, float r_sev, float r_seh);

/* Field order is fixed and documented once here; the definition (in
 * test_harness.cpp) and every Python parser must agree on this exact
 * order. All quantities are the surface node (i=1)'s own, for ONE
 * Newton sub-iteration, in the order the pinned code itself computes
 * them (fof_soi.cpp's per-node for-loop body, i==1 case). */
typedef struct {
    float old_tn1, new_tn1;             /*  0, 1: temperature before/after this sub-iteration's update */
    float old_p1, new_p1;               /*  2, 3: matric potential before/after */
    float old_wn1, new_wn1;             /*  4, 5: water content before/after (watercontent() output) */
    float old_h1, new_h1;               /*  6, 7: humidity before/after (humidity() output) */
    float psat0, h0;                    /*  8, 9: boundary (air) node vapor pressure/humidity, this sub-iter */
    float psat1;                        /* 10: surface saturation vapor pressure (from OLD tn1) */
    float psat2, h2;                    /* 11, 12: node-2 vapor pressure/humidity as seen by this sub-iter (still old, node 2 not yet updated this sweep) */
    float s1, hvap1;                    /* 13, 14: slope() and Hv() outputs at node 1, from OLD tn1 */
    float kh1, enh1, kv1;               /* 15-17: node-1 thermal cond./vapor-enhancement/vapor cond. (tcond/Kvap, from OLD state) */
    float kh2, kv2;                     /* 18, 19: node-2 thermal/vapor conductivity computed THIS sub-iteration */
    float ke0, ke1;                     /* 20, 21: effective (liquid+convective) conductance, boundary/node-1 */
    float kev0, kev1;                   /* 22, 23: effective vapor conductance, boundary/node-1 */
    float conv1, vcon1, cp1;            /* 24-26: convective term, vapor-convective term, heat capacity, node 1 */
    float d_jv, d_jvdt, d_jvdp;         /* 27-29: vapor-flux residual and its T/p derivatives */
    float dC_before_boundary;           /* 30: heat residual BEFORE the i==1 Stefan-Boltzmann correction */
    float dC_after_boundary;            /* 31: heat residual AFTER the correction (== r_dC used in the Newton solve) */
    float dv;                           /* 32: water-mass residual (never touched by the boundary correction) */
    float dCdp, dvdp;                   /* 33, 34: off-diagonal Jacobian terms */
    float dCdt_before_boundary;         /* 35: diagonal heat-Jacobian term BEFORE the correction */
    float dCdt_after_boundary;          /* 36: diagonal heat-Jacobian term AFTER the correction (== r_dCdt used) */
    float dvdt;                         /* 37: diagonal water-Jacobian term (never touched by the correction) */
    float r_rabs_in;                    /* 38: this timestep's total absorbed-radiation input (the function's own r_Rabs parameter) */
    float stefan_term;                  /* 39: 5.67e-8*r_tk*r_tk3, the outgoing-radiation term added by the correction */
    float tk_old;                       /* 40: surface temperature in Kelvin, from OLD tn1 (== r_tk) */
    float dtn_temperature_raw;          /* 41: the Newton increment for temperature, BEFORE the <-100 clamp */
    float dtn_temperature_clamped;      /* 42: 1.0 if the <-100 clamp fired this sub-iteration, else 0.0 */
    float dtn_matric_raw;               /* 43: the Newton increment for matric potential (no clamp of its own) */
    float p1_before_range_clamp;        /* 44: p[1] immediately after subtracting dtn_matric_raw, before the >0/<-1e20 checks */
    float p1_clamp_branch;              /* 45: 0.0 = no clamp, 1.0 = the p>0 halving branch fired, 2.0 = the <-1e20 floor fired */
    float r_sev_running, r_seh_running; /* 46, 47: cumulative |dv|/|dC| totals through node 1 (i.e. i=1's own contribution) */
} SoiSurfaceUpdateDiag;

void SoiDiagRecordSurfaceUpdate(int n_subiter, const SoiSurfaceUpdateDiag *d);
}

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.- */
#define   e_zmax     0.3   /* {depth of lower boundary, m                 } */
#define   e_Patm   92000   /* {atmospheric pressure at simulation site, Pa} */
#define   e_Dvo  2.12e-5   /* {vapor diffusivity in air, m2/s             } */
#define   e_Tstd  273.15   /* {standard temperature, K                    } */
#define   e_Po    101300   /* {sea level or standard pressure, Pa         } */
#define   e_R     8.3143   /* {gas constant, J/mol/K                      } */
#define   e_Mw     0.018   /* {mole mass of water, kg/mol                 } */
#define   e_hc        20   /* {surface boundary layer resistance          } */
#define   e_epse     100   /* {energy balance error - W/m2                } */
#define   e_epsw    1e-5   /* {water mass balance error - kg/(m2 s)       } */
#define   e_dw      1000   /* {density of water - kg/m3                   } */
#define   e_tor     0.66   /* {soil tortuosity - dimensionless            } */
#define          e_maxits    20   /* {maximum number of iterations in solution   } */
#define   e_airvp   1000   /* {air vapor pressure - Pascals               } */
#define   e_Tair      20   /* {starting air temp - C                      } */


/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.- */

FILE *fh_In;

float  r_seh, r_sev, r_wav,  r_tav,  r_dJv,  r_dJvdt, r_dJvdp;
float  r_bd,  r_pd,  r_ls,   r_ga,   r_xwo,  r_cop,   r_xo;
float  r_dC,  r_dv,  r_dCdp, r_dvdp, r_dvdt, r_dCdt,  r_m;
float  r_tk,  r_tk3, r_dtn,  r_gvol, r_ch,   r_xs,    r_xws; ;

float  rr_wn   [e_mplus1+1],  rr_w    [e_mplus1+1],  rr_z[e_mplus1+1];
float  rr_p    [e_mplus1+1],  rr_dwdp [e_mplus1+1],  rr_v[e_mplus1+1];
float  rr_h    [e_mplus1+1],  rr_tn   [e_mplus1+1],  rr_dhdp [e_mplus1+1];
float  rr_psat [e_mplus1+1],  rr_kev  [e_mplus1+1],  rr_u    [e_mplus1+1];
float  rr_Hvap [e_mplus1+1],  rr_s    [e_mplus1+1],  rr_ke   [e_mplus1+1];
float  rr_kh   [e_mplus1+1],  rr_kv   [e_mplus1+1],  rr_cp   [e_mplus1+1];
float  rr_conv [e_mplus1+1],  rr_vcon [e_mplus1+1],  rr_enh  [e_mplus1+1];
float  rr_t    [e_mplus1+1];

float  rr_AirPor[e_mplus1+1];


float  sPOW (float x, float y);


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: soiltemp_step
* Desc: This code was converted from orginal Pascal code, not really sure
*        what it exactly does
* Note-1: Change 11-6-05, put work around for bug found.
*         Code can get stuck in an infinite loop. Discovered this with
*          some of DL's batch data.
*         I checked back with the orginal Pascal code and verified that
*          I converted the code correctly which it looks like I did.
*         This function the and code that call it have some serious logic
*          errors, see the i_its variable below, it never gets incremented
*          but is checked in the loop as a break control. I tried implementing
*          it but didn't completly help as the calling functions loop
*          would keep calling it again and again, seems that calling loop
*          doesn't have a way of timing out.
*         The inputs being sent into the upper lever Soil Sim via the
*           d_SI input struct that cause this problem are, roughly speaking,
*           because they seem vary with ranges are approx. fi 89, time 180
*           moisture 25 'WET'
*         ER suggest we just put a check in to time out the loop and report
*          back to user that Soil Sim doesn't handle the situation
*         NOTE: I modified this function from a void to and int return,
*          and return 0 for the bug loop time out
* Ret: 1 ok, else 0 = error, read above
*
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
int soiltemp_step (float r_Rabs, float r_dt,  int  *ai_success)
{

int i,i_its;
int iN_SoilBug;

   iN_SoilBug = 0;
   if ( r_dt <= 0 ) 
     return 0; 

/* float fff; */
   rr_tn[0] = (float)e_Tair;
   i_its = 0;

/*.......................................................................    */
   while (1) {                    /* {begin heat and water solutions}       */
     r_seh = 0;
     r_sev = 0;
     rr_ke[0] = 0;                      /* {hc}; {hc taken out so that surf. temp. not needed} */
     rr_kev[0] = 6.2e-9 * (float)e_hc;  /* {6.2e-9*hc makes heat and water cond. equal} */
     rr_psat[0] = vaporpressure (rr_tn[0]);
     rr_h[0] = (float)e_airvp / rr_psat[0];
     rr_psat[1] = vaporpressure (rr_tn[1]);
     r_wav = 0.5 * (rr_wn[1] + rr_wn[2]);
     rr_s[1] = slope (rr_tn[1], rr_psat[1]);
     rr_Hvap[1] = Hv (rr_tn[1]);
     rr_kh[1] = tcond (rr_tn[1], r_wav, r_xs, r_ls, r_ga, r_xwo, r_cop,  rr_h[1] * rr_psat[1], rr_s[1], &rr_enh[1]);
     rr_AirPor[1] = (r_xws - r_wav);
     rr_kv[1] = rr_enh[1] * rr_AirPor[1] * (float)e_tor * Kvap(rr_t[1], rr_psat[1] * rr_h[1]);

/* F-70 diagnostic snapshot: node 1's state as it stands BEFORE this
 * sub-iteration's own update -- nothing above this point in the while
 * loop ever writes rr_tn[1]/rr_p[1]/rr_wn[1]/rr_h[1], so this is a pure
 * read, not a reordering of any pinned statement. */
float diag_old_tn1 = rr_tn[1];
float diag_old_p1 = rr_p[1];
float diag_old_wn1 = rr_wn[1];
float diag_old_h1 = rr_h[1];

/*..........................................................................*/
     for ( i = 1; i <= r_m; i++ ) {
       rr_cp[i] = rr_v[i] * (0.87 * r_bd + 4.18e6 * rr_wn[i]) / r_dt;
       rr_psat[i+1] = vaporpressure (rr_tn[i+1]);
       if ( i < r_m ) {
         r_wav = 0.5 * (rr_wn[i+1] + rr_wn[i+2]);
         r_tav = 0.5 * (rr_tn[i+1] + rr_tn[i+2]) + 273; }
       else {
           r_wav = rr_wn[e_mplus1];
           r_tav = rr_tn[i+1] + 273; }
       rr_conv[i] = 0.5 * (rr_u[i-1] + rr_u[i]) * 1200 * 293 / r_tav;
       rr_vcon[i] = rr_conv[i] * (float)e_Mw / ( (float)e_R * 1200 * 293);
       rr_s[i+1] = slope(rr_tn[i+1], rr_psat[i+1]);
       rr_Hvap[i+1] = Hv (rr_tn[i+1]);
       rr_kh[i+1] = tcond (rr_tn[i+1], r_wav, r_xs, r_ls, r_ga, r_xwo, r_cop, rr_h[i+1] * rr_psat[i+1], rr_s[i+1], &rr_enh[i+1]);
       rr_ke[i] = (rr_kh[i]) / ((rr_z[i+1] - rr_z[i])) + rr_conv[i];
       rr_AirPor[i+1] = (r_xws - r_wav);
       rr_kv[i+1] = rr_enh[i+1] * rr_AirPor[i+1] * (float)e_tor * Kvap (rr_t[i+1], rr_psat[i+1] * rr_h[i+1]);
       rr_kev[i] = (rr_kv[i] + rr_kv[i+1]) / (2 * (rr_z[i+1] - rr_z[i])) + rr_vcon[i];
       r_dJv = rr_kev[i-1] * (rr_psat[i] * rr_h[i] - rr_psat[i-1] * rr_h[i-1]) - rr_kev[i] * (rr_psat[i+1] * rr_h[i+1] - rr_psat[i] * rr_h[i]);
       r_dJvdt = rr_s[i] * rr_h[i] * (rr_kev[i-1] + rr_kev[i]);
       r_dJvdp = rr_psat[i] * (rr_kev[i-1] + rr_kev[i]) * rr_dhdp[i];
       r_dC =   rr_ke[i-1] * (rr_tn[i] - rr_tn[i-1])
              - rr_ke[i] * (rr_tn[i+1] - rr_tn[i])
              + rr_cp[i] * (rr_tn[i] - rr_t[i])
              - rr_Hvap[i] * (float)e_dw * rr_v[i]
              * (rr_wn[i] - rr_w[i]) / r_dt;
       r_dv = r_dJv + (float)e_dw * rr_v[i] * (rr_wn[i] - rr_w[i]) / r_dt;
       r_dCdp = - rr_Hvap[i] * (float)e_dw * rr_v[i] * rr_dwdp[i] / r_dt;
       r_dvdp = r_dJvdp + (float)e_dw * rr_v[i] * rr_dwdp[i] / r_dt;
       r_dvdt = r_dJvdt;
       r_dCdt = rr_ke[i] + rr_ke[i-1] + rr_cp[i];
       /* F-70 diagnostic snapshot (i==1 only): capture the residual
        * numerator/Jacobian pair BEFORE the boundary correction below
        * can touch them -- pure reads, no reordering. */
       float diag_dC_before = r_dC;
       float diag_dCdt_before = r_dCdt;
       float diag_stefan_term = 0.0f;
       float diag_tk_old = 0.0f;
       if ( i == 1) {
         r_tk = rr_tn[1] + 273;
         r_tk3 = r_tk * r_tk * r_tk;
         diag_stefan_term = 5.67e-8 * r_tk * r_tk3;
         diag_tk_old = r_tk;
         r_dC = r_dC - r_Rabs + 5.67e-8 * r_tk * r_tk3;
         r_dCdt = r_dCdt + 4 * 5.67e-8 * r_tk3; }

       r_sev = r_sev + abs_Real(r_dv);
       r_seh = r_seh + abs_Real(r_dC);

       r_dtn = (r_dv * r_dCdp - r_dC * r_dvdp) / (r_dCdp * r_dvdt - r_dCdt * r_dvdp);
       /* F-70 diagnostic snapshot: the raw temperature increment and
        * whether the <-100 clamp fires, BEFORE the clamp -- pure read. */
       float diag_dtn_temp_raw = r_dtn;
       float diag_dtn_temp_clamped = (r_dtn < -100) ? 1.0f : 0.0f;
       if ( r_dtn < -100 )
         r_dtn = -100;
       rr_tn[i] = rr_tn[i] - r_dtn;
       r_dtn = ( r_dv - r_dvdt * r_dtn) / r_dvdp;
       float diag_dtn_matric_raw = r_dtn;
       rr_p[i] = rr_p[i] - r_dtn;
       float diag_p_before_range_clamp = rr_p[i];
       float diag_p_clamp_branch = 0.0f;
       if ( rr_p[i] > 0 ) {
         rr_p[i] = ( rr_p[i] + r_dtn ) * 0.5;
         diag_p_clamp_branch = 1.0f; }
       if ( rr_p[i] < -1e20 ) {
         rr_p[i] = -1e20;
         diag_p_clamp_branch = 2.0f; }
       rr_wn[i] = watercontent (rr_p[i], r_xo, &rr_dwdp[i]);
       rr_h[i] = humidity (rr_p[i], rr_tn[i], &rr_dhdp[i]);
       if ( i == 1 ) {
         /* F-70 diagnostic hook (third addition, third same-pass round):
          * fires once per Newton sub-iteration, capturing every named
          * quantity the surface-node update itself just read or wrote
          * this sub-iteration -- see SoiSurfaceUpdateDiag's own field
          * comments (declared in fof_soi_instr.cpp's own header block)
          * for the exact meaning of each field. */
         SoiSurfaceUpdateDiag diag;
         diag.old_tn1 = diag_old_tn1;             diag.new_tn1 = rr_tn[1];
         diag.old_p1 = diag_old_p1;               diag.new_p1 = rr_p[1];
         diag.old_wn1 = diag_old_wn1;             diag.new_wn1 = rr_wn[1];
         diag.old_h1 = diag_old_h1;               diag.new_h1 = rr_h[1];
         diag.psat0 = rr_psat[0];                 diag.h0 = rr_h[0];
         diag.psat1 = rr_psat[1];
         diag.psat2 = rr_psat[2];                 diag.h2 = rr_h[2];
         diag.s1 = rr_s[1];                       diag.hvap1 = rr_Hvap[1];
         diag.kh1 = rr_kh[1];                     diag.enh1 = rr_enh[1];
         diag.kv1 = rr_kv[1];
         diag.kh2 = rr_kh[2];                     diag.kv2 = rr_kv[2];
         diag.ke0 = rr_ke[0];                     diag.ke1 = rr_ke[1];
         diag.kev0 = rr_kev[0];                   diag.kev1 = rr_kev[1];
         diag.conv1 = rr_conv[1];                 diag.vcon1 = rr_vcon[1];
         diag.cp1 = rr_cp[1];
         diag.d_jv = r_dJv;                       diag.d_jvdt = r_dJvdt;
         diag.d_jvdp = r_dJvdp;
         diag.dC_before_boundary = diag_dC_before;
         diag.dC_after_boundary = r_dC;
         diag.dv = r_dv;
         diag.dCdp = r_dCdp;                      diag.dvdp = r_dvdp;
         diag.dCdt_before_boundary = diag_dCdt_before;
         diag.dCdt_after_boundary = r_dCdt;
         diag.dvdt = r_dvdt;
         diag.r_rabs_in = r_Rabs;
         diag.stefan_term = diag_stefan_term;
         diag.tk_old = diag_tk_old;
         diag.dtn_temperature_raw = diag_dtn_temp_raw;
         diag.dtn_temperature_clamped = diag_dtn_temp_clamped;
         diag.dtn_matric_raw = diag_dtn_matric_raw;
         diag.p1_before_range_clamp = diag_p_before_range_clamp;
         diag.p1_clamp_branch = diag_p_clamp_branch;
         diag.r_sev_running = r_sev;              diag.r_seh_running = r_seh;
         SoiDiagRecordSurfaceUpdate(iN_SoilBug + 1, &diag);
       }
     } /* for i end */

/*...........................................................................*/

/* Change 11-6-05, Catch infinite loop bug, See Note-1 above                 */
/* Change 4-2-15, take this check out, see the fix where we make the call to this function */
     iN_SoilBug++;

     /* F-70 DIAGNOSTIC HOOK (second addition, same pass): fires once PER
      * Newton sub-iteration, right after r_sev/r_seh are finalised for
      * this sweep and BEFORE the convergence break-check below -- gives
      * the Python-side comparison the per-sub-iteration trajectory
      * (r_sev, r_seh, node-1 tn/p) that the end-of-step hook alone cannot
      * (it only ever sees the FINAL converged sweep). Surface-node (i=1)
      * values only -- sufficient to locate the first diverging
      * sub-iteration without a second full per-node array dump. */
     SoiDiagRecordSubIteration(iN_SoilBug, rr_tn[1], rr_p[1], r_sev, r_seh);

     if ( iN_SoilBug >= 500)
         return 0;
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

     if ( (r_sev < (float)e_epsw  && r_seh < (float)e_epse) || i_its > e_maxits )
       break;

    } /* while (1) end */

/*.........................................................................*/
   if ( i_its < e_maxits ) {
     *ai_success = 1;

/* changed 3-27-00 */
/*     rr_u[r_m] = 0;    got error when compliling as C++ */
       rr_u[(int)r_m] = 0;


     for ( i = r_m; i >= 1; i-- ) {
       r_gvol = (float)e_dw * rr_v[i] * (float)e_R * (rr_tn[i] + 273) * (rr_w[i] - rr_wn[i]) / (r_dt * rr_AirPor[i] * (float)e_Mw * (float)e_Patm);
       if ( r_gvol < 0 )
         r_gvol = 0;
       rr_u[i-1] = rr_u[i] + r_gvol; }
     for ( i = 1; i <= e_mplus1; i++ ) {
       r_ch = rr_wn[i] - rr_w[i];
       rr_w[i] = rr_wn[i];
       rr_wn[i] = rr_w[i] + r_ch;
       r_ch = rr_tn[i] - rr_t[i];
       rr_t[i] = rr_tn[i];
       rr_tn[i] = rr_t[i] + r_ch; }

     /* F-70 DIAGNOSTIC HOOK -- the ONLY line added to this instrumented
      * copy relative to the real pinned fof_soi.cpp (see the file-header
      * comment above). Called exactly once per successfully-converged
      * outer timestep, right after the real commit/extrapolation step
      * above and before this function's own `return 1;` -- so `rr_t[]`
      * here is the TRUE Newton-converged value for this timestep (just
      * committed), `rr_tn[]`/`rr_wn[]` are the EXTRAPOLATED predictor
      * values SD_Mngr_New's/SE_Mngr_Array's own soiltemp_gettemps()/
      * soiltemp_getwater() will read and SHA_Put() will actually save
      * (see F-70's prior diagnostic pass), and rr_p/rr_h/rr_psat/rr_kh/
      * rr_kv/rr_enh are untouched by the commit step so are still the
      * real converged values from this timestep's last Newton
      * sub-iteration. iN_SoilBug (renamed n_subiter in the hook) is the
      * real sub-iteration count this timestep actually took. */
     SoiDiagRecordTimestep(e_mplus1 + 1, rr_tn, rr_t, rr_wn, rr_w, rr_p, rr_h,
                           rr_psat, rr_kh, rr_kv, rr_enh, r_sev, r_seh,
                           iN_SoilBug, r_Rabs);
   }
   else
     *ai_success = 0;

   return 1;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: tcond
* Desc:
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
float   tcond (float r_t, float r_xw, float r_xs, float r_ls, float r_ga,
              float r_xwo, float r_cop, float r_p, float r_s, float *ar_enh)
{
/*  float  f; */
float  r_wf,r_ka,r_ks,r_kw,r_la,r_lw,r_lf,r_xa,r_xws,r_gc,r_lda,r_xv,r_tc;
/*  float  g,h; */
float  r_A, r_B, r_C;
   r_xws = 1 - r_xs;
   r_xa = r_xws - r_xw;
   if ( r_t < 100 ) {
     r_lw = 0.554 + r_t * (2.24e-3 - 9.87e-6 * r_t);
     r_tc = sqr (sqr( (r_t + 273) / 303)); }
   else {
     r_lw = 0.68;
     r_tc = 2.3; }
   r_lda = 0.024 +  r_t * (7.73e-5 - 2.6e-8 *  r_t);
   if ( r_xw < (0.01 * r_xwo) )
     r_wf = 0;
   else
     r_wf = 1 / (1 + sPOW(r_xw / r_xwo,-r_cop * r_tc));

/*...orginal line  r_la = r_lda + r_wf * Hv(r_t) * r_s * Kvap(r_t,r_p);      */
   r_A = Kvap(r_t,r_p);
   r_B = Hv(r_t);
   r_C = ( r_wf * r_B * r_s * r_A);
   r_la = r_lda + r_C ;

   r_gc = 1 - 2 * r_ga;
   r_lf = r_la + (r_lw - r_la) * r_wf;
   r_ka = (2 / (1 + (r_la / r_lf -1) * r_ga) + 1 / (1 + (r_la / r_lf - 1) * r_gc ) ) / 3;
   r_kw = (2 / (1 +(r_lw / r_lf - 1) * r_ga) + 1 / (1 + (r_lw / r_lf - 1) * r_gc )) /3;
   r_ks = (2 / (1 +(r_ls / r_lf - 1) * r_ga) + 1 / (1 + (r_ls / r_lf - 1) * r_gc )) /3;
  *ar_enh = ( 1 + 2 * r_wf) * r_ka;
   r_tc = (r_kw * r_lw * r_xw + r_ka * r_la * r_xa + r_ks * r_ls * r_xs) / (r_kw * r_xw + r_ka * r_xa + r_ks * r_xs);
   if ( r_tc > 0  &&  r_tc < 5)
     return r_tc;
   return 1;
}



/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name:
* Desc:
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
void soiltemp_initconsts (float r_bdi, float r_pdi, float r_lsi, float r_gai,
                     float r_xwoi, float r_copi, float r_xoi,
                     float rr_zi[] )
{
int  i;

   r_bd  = r_bdi;
   r_pd  = r_pdi;
   r_ls  = r_lsi;
   r_ga  = r_gai;
   r_xwo = r_xwoi;
   r_cop = r_copi;
   r_xo  = r_xoi;
   r_m   = e_mplus1 - 1 ;
   r_xs  = r_bd / r_pd;
   r_xws = 1 - r_xs;

   for ( i = 0; i <= e_mplus1; i++ ) {
     if ( rr_zi[i] < 0 )
       break;
     rr_z[i] = rr_zi[i];
     rr_z[i] = rr_z[i] / 1000;
   }

/*........................................*/

   for ( i = 0; i <= e_mplus1; i++  )
     rr_v[i] = 0;
   for ( i = 1; i <= r_m; i++  )
     rr_v[i] = 0.5 * ( rr_z[i+1] - rr_z[i-1] );
   rr_AirPor[0] = 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name:
* Desc:
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
void   soiltemp_initprofile (float rr_wi[], float rr_ti[])
{
int  i;

   Copy_Array ( rr_w,  rr_wi);
   Copy_Array ( rr_wn, rr_wi);
   Copy_Array ( rr_t,  rr_ti);
   Copy_Array ( rr_tn, rr_ti);
   for ( i = 0; i <= e_mplus1; i++ )  {
     rr_p[i] = -exp ( 13.82 * ( 1 - rr_w[i] / r_xo ) );
     rr_w[i] = watercontent ( rr_p[i], r_xo, &rr_dwdp[i]);
     rr_h[i] = humidity ( rr_p[i], rr_t[i], &rr_dhdp[i] );
     rr_kev[i] = 0;
     rr_u[i] = 0;
     rr_enh[i] = 0; }
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: watercontent
* Desc:
* Note-1: This log(0) never happened but figured I'd better put a check in
*         in case it ever does. I think trying to do a log(0) will blowup
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
float watercontent (float r_p, float r_xo, float *ar_dwdp)
{
#define e_lnpo  13.82         /* ln of oven dry water content */
float f;

  if ( r_p >= 0 )
     r_p = -0.001;
  *ar_dwdp = -r_xo / ( (float)e_lnpo * r_p );

  if ( r_p == 0 ) {                     /* See Note-1 above                  */
    strcpy (gcr_SoiErr, "watercontent() - Math Error - attempted to do a log(0)");
    r_p = 1; }

  f =  r_xo * ( 1 - log(-r_p) / (float)e_lnpo);
  return f;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: humidity
* Desc:
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
float  humidity (float r_p, float r_t,  float *ar_dhdp)

{
float r_h, r_Tk;
  r_Tk = r_t + (float)e_Tstd;
  r_h = exp( (float)e_Mw * r_p / ( (float)e_R * r_Tk));
  *ar_dhdp = (float)e_Mw * r_h / ((float)e_R * r_Tk);
  return r_h;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: vaporpressure
* Desc:
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
float   vaporpressure (float r_tin)
{
float r0,r1,r2,r3, r_t;
  r_t = r_tin * 1000;
  r0 =  ( r_t + 273150);
  r1 = 373150 / r0;
  r_t = 1 - r1;
/*  r_t = 1 - 373.15 / ( r_t + 273.15); */

  r1 = r_t * (13.3016 + r_t * (-2.042 + r_t * (0.26 + r_t * 2.69)));
  r2 = exp (r1);
  r3 = 101325 * r2;
  return r3;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: slope
* Desc:
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
float   slope  (float r_t, float r_p)
{
float r_dydt,r_Tk, f;
   r_Tk = r_t + (float)e_Tstd;
   r_t = 1 - 373.15 / r_Tk;
   r_dydt = 373.15 / sqr(r_Tk);
   f = r_p * r_dydt * (13.3015 + r_t * (-4.082 + r_t * (0.78 + r_t * 10.76) ));
   return f;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name:
* Desc: returns latent heat of vaporization in J/kg
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
float   Hv  (float r_t)
{
 return ( 2.508e6 - 2670 * r_t );
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Kvap
* Desc: returns vapor conductivity in kg/(m s Pa)
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
float  Kvap (float r_t, float r_p)
{
float r_Tk, r_Dv, r_stcor, f, g;
   r_Tk = r_t + (float)e_Tstd;
   f = sPOW (r_Tk / (float)e_Tstd, 1.75);
   g = ((float)e_Po / (float)e_Patm);
   r_Dv =  (float)e_Dvo * g * f ;
   r_stcor = 1 - r_p / (float)e_Patm;
   if ( r_stcor < 0.3 )
     r_stcor = 0.3;
   f = (float)e_Mw * r_Dv / ( (float)e_R * r_Tk * r_stcor);
   return f;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: sPOW
* Desc: the pascal code use the lower case 'pow' which was a function
*        function right in the pascal code, not a math lib function
*        so I re-did it here and made it upper case 'POW' so it doesn't
*        confict with the C math libary one.
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
float  sPOW (float x, float y)
{
float f,g,h;
  if ( x < 0 )
    x = -x;

  if ( x == 0 )
    f = 0;
  else {
    g = log(x);
    h = y * g;
    f = exp ( h ); }

  return f;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name:
* Desc:
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
void soiltemp_getwater ( float rr_wi[])
{
   Copy_Array (rr_wi,rr_wn);
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name:
* Desc:
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
void soiltemp_gettemps ( float rr_ti[])
{
   Copy_Array (rr_ti,rr_tn);
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name:
* Desc:
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
void soiltemp_getdepths ( float rr_zi[])
{
 Copy_Array (rr_zi,rr_z);
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: sqr
* Desc:
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
float  sqr  (float  r)
{
  return r * r;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name:
* Desc:
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
float abs_Real (float r)
{
  if ( r < 0 )
    return (r * -1);
  return r;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name:
* Desc:
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
void Copy_Array ( float rr_to[], float rr_from[])
{
int i;
   for ( i = 0; i <= e_mplus1; i++  )
      rr_to[i] = rr_from[i];
}


#ifdef wow
/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name:
* Desc:
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
void Display_Array ( float rr[])
{
int i;
  printf ("------------------------------------------\n");
   for ( i = 0; i <= e_mplus1; i++  )
    printf ("%d - %15.6f \n", i, rr[i]);
  printf ("------------------------------------------\n");
}
#endif
