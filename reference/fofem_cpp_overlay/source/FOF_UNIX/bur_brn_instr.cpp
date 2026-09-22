 /*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: bur_brn.c     Burnup - Actual Simulation
* Desc: The actual simulation is done is this module
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#include  "fof_gen.h"
#include  "bur_brn.h"
#include  "fof_util.h"
#include  "fof_sgv.h"
#include  "bur_bov.h"
#include  "fof_lem.h"
#include "brn_filesw.h"
#include "unit_abv.h"
#include "fof_ansi.h"

#include "..\config.h"


// test...........................
float gfN_FlaCon = 0; 
float gfN_SmoCon = 0; 

// .............................

double FIC_Percent ();
void FIC_Init ();
double FIC_SmoPercent (double d_SURat);
double FIC_1kSmoPercent ();
bool _isEqual (double a, double b);
void FIC_Put (double d_SURat, double d_dendry, double term, double test, double amount, char cr_FS[]);
void _CompDump (float test, float cmpark, float term, char FS[], float wnoduff, double sigma);

int ES_FlaSmo (d_ES *a, float *af_Smo, float *af_Fla, float *af_1kSmo, float *af_1kFla);

/*..................................................*/
/* Testing for FOFEM */
#ifdef FOFEM
char cr_DuffType[100] = "";  
float   _FireInt_DuffType  (double Rm);
#endif 
 
/*...................................................*/

double  dr_Done[MAXNO]; 
double  dr_DD[MAXNO]; 

/* Used to signal that the Emission Output file (when done) should output */
/*  Flamming and Smoldering values */
char gcr_EFFS[30];

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
long    ntimes, number;
double  fi, ti, u, d, tamb, ak, r0, dr, dt, wdf, dfm;
double  wd0, wg0;
double  wdry [MAXNO], ash   [MAXNO], htval[MAXNO];
double  fmois[MAXNO], dendry[MAXNO], sigma[MAXNO];
double  cheat[MAXNO], condry[MAXNO], alfa[MAXNO];
double  tpig [MAXNO], tchar [MAXNO];
double  flit [MAXNO], fout  [MAXNO], work[MAXNO];
long    key[MAXNO];

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
double     elam [MAXNO][MAXNO];
double     alone[MAXNO];
double     area [MAXNO];
double     fint [MAXNO];
double     xmat [MAXKL];
double     tdry [MAXKL];
double     tign [MAXKL];
double     tout [MAXKL];

// 'wo' was the original name of the array, but debugger wouldn't display
// it with a 'watch' for whatever reason, so I changed the name here and 
// throughout the code and then the debugger let me 'watch' it 
// double     wo   [MAXKL];
double     WOO  [MAXKL];   /* test */

double     wodot[MAXKL];
double     diam [MAXKL];
double     ddot [MAXKL];
double     qcum [MAXKL];
double     tcum [MAXKL];
double     acum [MAXKL];
double     qdot [MAXKL][MXSTEP];

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
static const double ch2o = 4186.0;
static const double tpdry = 353.0;

double gd_Fudge1;
double gd_Fudge2;

d_ES   s_ES;                            /* Emission Structure                */

/* Emission Factors - Flame and Smoldering */ 
float gf_fCO, gf_fCO2, gf_fCH4, gf_fPM25, gf_fPM10, gf_fNOX, gf_fSO2; /* Flame */
float gf_sCO, gf_sCO2, gf_sCH4, gf_sPM25, gf_sPM10, gf_sNOX, gf_sSO2; /* Smolder Coarse Wood */
float gf_dCO, gf_dCO2, gf_dCH4, gf_dPM25, gf_dPM10, gf_dNOX, gf_dSO2; /* Duff */ 

/* Critical Fire Intensity point which dictates if Flame or Smold */
/*  factor will get used */
float gf_CriInt;

/*-----------------------------------------------------------*/
/* Original - Emission factors                               */
/* 5-11-2016 Took these out of the ES_Calc() so that I could */
/* get at them and pass them back to the FuelCalc GUI       */ 
double   d_pm25f  = 67.4 - e_ComEffFla *  66.8;    /* Emission Factors             */
double   d_pm25s  = 67.4 - e_ComEffSmo *  66.8;    /* NOTE, This gets them to      */
double   d_ch4f  = 42.7 - e_ComEffFla *  43.2;     /* Grams Per Square Meter       */
double   d_ch4s   = 42.7 - e_ComEffSmo *  43.2;
double   d_cof    = 961  - e_ComEffFla * 984.0;
double   d_cos    = 961  - e_ComEffSmo * 984.0;
double   d_co2f   = e_ComEffFla * 1833.0;
double   d_co2s   = e_ComEffSmo * 1833.0;
double   d_pm10f  = d_pm25f * 1.18;
double   d_pm10s  = d_pm25s * 1.18;
double   d_noxf =  3.2;                  /* NOX & SO2 emission factors             */
double   d_noxs =  0;
double   d_so2f =  1.0;
double   d_so2s =  1.0;

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Min and Max won't port so I but defined and put the functions in here     */
double  Min (double a, double b);
double  Max (double a, double b);

int _ChkLimM (double d, double low, double hi);
int  ES_Calc_NEW (d_ES *a_ES, double d_WooLit,  double d_Duff, double d_HSFB, double d_time, float f_FirInt);
double  ES_SetComponents (int iS_Init, float *af_TPA, double d_Time, d_ES *a_ES);
int ES_SetComp (int i, double d_Rem, double d_Con, d_ES *a_ES );

/* Grand Fuel Consumed Total accumulated only when doing Expanded Emission */ 
/* see  ES_Calc_NEW()  */
double gdN_FlameCon = 0;  
double gdN_SmolderCon = 0; 
double gdN_DuffCon = 0;   

/**********************************************************************/
/* Soil Heating Fire Intensity Array                                  */
/* Store fire intensity to be used by soil heating model              */
/* each position of the array is 15 seconds                           */
/* These fire intensity values are used to do the non-duff soil       */
/* so we only store the fire intensity for wood liter herb shrub      */
#define eC_SFI 6000   /* Soil Heating fire intensity array */
float  fr_SFI[eC_SFI + 1 ];
int    iX_SFI;

float  fr_SFIhs[eC_SFI + 1 ];

/*-----------------------------------------------------
* Init the array and put the first 60 second fire 
* intensity into the first four positions
* Note-1: For now I'm going to assume that the fire 
*  ramps up to it full rate across the first 60 seconds
* Also see notes above 
*------------------------------------------------------*/
void SFI_Init (float f_WLfi, float f_HSfi) 
{
int i; 
float f; 
   for ( i = 0; i <= eC_SFI; i++ ) {
      fr_SFI[i] = -1.0; 
      fr_SFIhs[i] = 1.0; }

/* See Note-1 above*/
   fr_SFI[0] = 0.25 * f_WLfi; 
   fr_SFI[1] = 0.50 * f_WLfi; 
   fr_SFI[2] = 0.75 * f_WLfi; 
   fr_SFI[3] = 1.0  * f_WLfi; 

   fr_SFIhs[0] = 0.25 * f_HSfi; 
   fr_SFIhs[1] = 0.50 * f_HSfi; 
   fr_SFIhs[2] = 0.75 * f_HSfi; 
   fr_SFIhs[3] = 1.0  * f_HSfi; 
   iX_SFI = 4; 
}

/***********************************************************/
void SFI_Put (float f_WLfi, float f_HSfi)
{
 if ( iX_SFI >= eC_SFI )
    return; 
  fr_SFI[iX_SFI] = f_WLfi;     /* Wood Litter fire intensity */
  fr_SFIhs[iX_SFI] = f_HSfi;   /* Herb Shrub fire intensity */
  iX_SFI++;
}

/********************************************************************
*
*
********************************************************************/
double  gd_TotHS;
double gd_PerSec;

double HSB_Get (double  d_Sec);
void  HSB_Init (float f_TotHS, float f_PerMin);

/***************************************************************
* Name: HSB_Init
* Desc: Given the amount of Herb + Shrub to be consumed per minute
*       calc the per second amount
*   In: f_TotHS... Total amount of herb + shrub in tons per acre
*       f_PerMin.. tons to be consumed per minute
***************************************************************/
void  HSB_Init (float f_TotHS, float f_PerMin)
{
   gd_PerSec = f_PerMin / 60.0;
   gd_TotHS = f_TotHS;  
}

/*************************************************************
*
*
*************************************************************/
double HSB_Get (double  d_Sec)
{
double d;
  d = gd_PerSec * d_Sec; 
  if ( d > gd_TotHS )
    d = gd_TotHS;
  gd_TotHS = gd_TotHS - d;
  return d; 
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: BRN_Run
* Desc: Run burnup.
* Note-1: a negative value in gf_CriInt set by the caller indicates that 
*         the user wants to do the emissions the old(original) way, otherwise
*         a positve value will cause the emissions to be done the new way with 
*         the new factors. 
*         When doing the old way the emission factors are already set here, 
*         but when doing the new way the caller needs to supply them, see
*         the BRN_SetEmis() function
*
*   In: cr_LoadFN.........name of output Post Load (remaining load)
*                         if empty string then no file will be created
*       cr_EmisFN........name of output Emissions File, if the is an empty
*                         string then no file will be created.
*       f_DufConPerCent..Percent of Duff Consumed
*                         Send in as a whole number 0 -> 100
*                         NOTE: send in a -1 and this will calc duff the
*                         old way with a defualt
*       f_ConHSFB........total Tons Per Acre consumed of Herb, Shrub,
*                        Foliage and Crown Branch,
*                        NOTE: send in 0 when running from a 'user' input
*                        file, because there is no calculated Herb,Shr,Fol,Bra
*  Ret: 1 - OK
*       2 - burnup can't ignite
*       0 - Error, the message is in the object
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int  BRN_Run (char cr_LoadFN[], char cr_EmisFN[], float f_DufConPerCent,
              float f_ConHerb, float f_ConShrb, float f_ConBraFol, char cr_ErrMes[])
{
int i, i_Ret;
long nruns = 0;
long now;
double fimin = 0.1;
double d, d_Time, d_DFI, d_tdf, d_1FI;
double d_ConWooLit,d_pcSmoCon;
double d_DufCon, d_Duf_Tot, d_Duf_Sec;
double d_HSFB_KiSq;
float  f;
double d_HSt; 
double d_HSfi, d_HSkg, d_FBfi; 

   _CompDump (0,0, 0, "Open", 0, 0);  /* Put out a test file - see _CompDump() */ 

   strcpy (cr_ErrMes,"");
   i_Ret = 1;

   d =  f_ConHerb + f_ConShrb;
   HSB_Init (d, 10.0 );                    /* store, we'll get back as needed */
   SGV_Init ();                            /* Init table, for Smoke saving   */
   ES_Init (&s_ES);                        /* Init Emission Sturct         */

   if ( !EFM_Open (cr_EmisFN) ) {          /* Open Emiss File if have nam  */
     strcpy (cr_ErrMes,"Can't Open Output Time Step Emissions File");
     i_Ret = 0 ;                           /* return value                 */
     goto X; }

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.- */
   if ( ntimes == 0 || number == 0) {
     strcpy (cr_ErrMes, "ntimes == 0 || number == 0");
     i_Ret = 0;
     goto X; }

   Arrays();                                 /* set loads and input vars     */
   now = 1;
   d_Time = ti;                              /* residual time ignition pulse */
   DuffBurn (wdf,dfm,&d_DFI,&d_tdf,f_DufConPerCent,&d_Duf_Sec);
   d_Duf_Tot = d_Duf_Sec * d_tdf;            /* Tot Amt of duff that will burn    */

// F-62 completion/acceptance-recovery pass (2026-09-21, second pass)
// DIAGNOSTIC HOOK. Same opt-in env var as the other hooks in this file --
// dumps DuffBurn()'s own inputs/outputs (wdf, dfm, d_tdf, d_Duf_Sec,
// d_Duf_Tot, ti) exactly once per BRN_Run invocation. Purely
// observational -- reads only, changes nothing.
if ( getenv("FOFEM_TEST_BURN_DIAG") != NULL ) {
  FILE *fhDB = fopen ("_DuffBurnDump.csv", "a");
  if ( fhDB != NULL ) {
    fprintf (fhDB, "wdf,dfm,d_DFI,d_tdf,f_DufConPerCent,d_Duf_Sec,d_Duf_Tot,ti\n");
    fprintf (fhDB, "%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
             wdf, dfm, d_DFI, d_tdf, f_DufConPerCent, d_Duf_Sec, d_Duf_Tot, ti);
    fclose (fhDB);
  }
}

   ES_SetComponents (1,&f,0,&s_ES);                   /* Init for collecting time step info */ 

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Do the First Time step - 60 Seconds                                       */
   i = Start(d_Time,now,&nruns,&d_ConWooLit);
   if ( i !=0 ) {
     i_Ret = 2;                              /* burnup can't ignite          */
     goto X; }

   fi = FireIntensity(&d_pcSmoCon);                     /* get Fire Intensity           */
   d_DufCon = Duff_CPTS (&d_Duf_Tot, d_Duf_Sec, 60.0);  /* duff that burn this time step*/
                  
   d_HSt = HSB_Get (60.0);                 /* Herb Shr consumed in first 60 seconds*/
   d_HSfi = BRN_Intensity(d_HSt,60.0);     /* Add Fire Int, Herb,Shr   */
   d_FBfi = BRN_Intensity(f_ConBraFol,60.0); 

   d = TPA_To_KiSq (d_HSt); 
   d += TPA_To_KiSq (f_ConBraFol); 
 
   if ( gf_CriInt < 0 )
     ES_Calc (&s_ES,                         /* calc Emissions               */ 
              d_ConWooLit,                   /* consumed Wood & litter */            
              d_DufCon,                      /* consumed duff */
              d,                             /* consumed herb shrb fol bra*/
              d_pcSmoCon,                    /* percent smolder consume*/
              d_Time);

/* When this gets hooked up again we'll need to total up fire intensity to send in */
/*  wood liter herb shrub fol branch, maybe duff ? */
  d_1FI = fi + d_HSfi;

  if ( gf_CriInt >= 0 ) 
    ES_Calc_NEW (&s_ES,d_ConWooLit,               /* calc Emissions               */
                  d_DufCon, d,  
                  d_Time,d_1FI);   


   SFI_Init (fi,d_HSfi);                  /* Save just these for soil heating model */ 

   fi = fi + d_HSfi + d_FBfi;             /* fir inten all fuels */
   EFM_Write (&s_ES,d_Time,fi);           /* put a line to Emission File  */
   BOV_Set_TimHea(d_Time,fi);             /* Save for doing Soil Exp Heat */
   Save_SGV (&s_ES,d_Time,fi);            /* Save stuff needed to graph   */

   if ( d_Time > d_tdf)                      /* if past time that duff will  */
     d_DFI = 0;                              /*  burnout                     */

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Do Time Steps - 15 sec default, see global 'dt', e_TIMESTEP              */
   while ( now <= ntimes) {                  /* Max times steps to do        */

     Step(dt,d_Time,d_DFI,&nruns,&d_ConWooLit);/* Do the time step              */
     now++;                                  /* count time steps             */

     d_Time += dt;                           /* increment time - seconds     */
     if ( d_Time > d_tdf)                    /* if past time that duff will  */
        d_DFI = 0;                           /* burnout                      */

     fi = FireIntensity(&d_pcSmoCon);        /* Fire Intensity - Wood Liter       */
     d_HSt = HSB_Get (dt);                   /* herb shrb consumed this time step */
     d_HSfi = BRN_Intensity(d_HSt,dt);       /* Fire Intensity - Herb Shrub  */
     d_HSkg = TPA_To_KiSq (d_HSt);           /* Herb Shrb to Kilograms       */
     SFI_Put (fi,d_HSfi);                    /* save for Soil heating model */
     fi = fi + d_HSfi;                       /* fire inten - wood liter herb shrub */

     d_DufCon = Duff_CPTS (&d_Duf_Tot,d_Duf_Sec,15.0);  /* duff that burn this time step*/
     
     if ( gf_CriInt < 0 )    
        ES_Calc (&s_ES, d_ConWooLit,d_DufCon, d_HSkg, d_pcSmoCon,d_Time); /* Emissions */
     else 
        ES_Calc_NEW (&s_ES, d_ConWooLit,d_DufCon, d_HSkg, d_Time,fi);  

	 EFM_Write (&s_ES,d_Time,fi);            /* put a line to Emission File  */
     BOV_Set_TimHea(d_Time,fi);              /* Save for doing Soil Exp Heat */
     Save_SGV (&s_ES, d_Time,fi);            /* Save stuff needed to graph   */

     if ( fi <= fimin) {                     /* fire go - from Intensity     */
        if ( d_Duf_Tot != 0 )                /* Duff remains Smoldering      */
          continue;                          /* Duf tot keeps getting usd up */
        break; }  
   }  /* while */

// char AA[100];
// strcpy (AA, "C:\\Users\\jreardon\\My Documents\\jim-test.txt"); 
// i = Summary(d_Time,AA,cr_ErrMes);          /* Do the Consmd Amts file      */
   i = Summary(d_Time,cr_LoadFN,cr_ErrMes);          /* Do the Consmd Amts file      */
   if ( i == 0 ) {                            /* error opening output file    */
     i_Ret = 0;
     goto X; }

X:
   EFM_Close ();

/* If we did Expanded Emissions - save these final fuel consumed amounts */
/*  see  ES_Calc_NEW()  */
  if ( gf_CriInt >= 0 ) {
    gdN_FlameCon =  s_ES.dN_FlameCon;
    gdN_SmolderCon = s_ES.dN_SmolderCon;  
    gdN_DuffCon =   s_ES.dN_DuffCon; }


// File Dump for testing - see _CompDump() */
   _CompDump (0, 0, 0, "Close", 0, 0);
 
   return i_Ret ;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Start
* Desc: Initialize time varying quantities and set up work( k )
*       The diameter reduction rate of fuel component k is given
*       by the product of the rate of heat transfer to it, per
*       unit surface area, and the quantity work( k )
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int  Start(double dt, long now, long *ncalls, double *ad_Con)
{
long   k, l, kl, nlit;
double delm, heatk, r, tf, ts, thd, tx, factor;
double conwet, dia,  hb, en, cpwet, fac;
double hf, dryt, tsd, c, tigk, e, dtign, trt;
double aint, ddt, dnext, wnext, df;
const double rindef=1.e+30;

double X,Y,Z;

     *ad_Con = 0;

     gd_Fudge1 = 0;
     gd_Fudge2 = 0;

     for ( k = 1; k <= number; k++) {            /* do k = 1 , number        */
        fout[k-1] = 0.0;
        flit[k-1] = 0.0;
        alfa[k-1] = condry[k-1] / (dendry[k-1] * cheat[k-1]);

/*    effect of moisture content on burning rate (scale factor) */
        delm=1.67* fmois[k-1];
/*    effect of component mass density (empirical) */
        heatk=dendry[k-1] / 446.0;
/*    empirical burn rate factor, J / cu m - K */
        heatk=heatk*2.01e+06*(1.0+delm);
/*    normalize out driving temperature difference (Tfire - Tchar) */
/*    to average value of lab experiments used to find above constants */
        work[k-1] = 1.0 / (255.0 * heatk);
        for ( l = 0; l <= k; l++) {                     /* do l = 0 , k */
           kl = loc(k,l);
           tout[kl] = rindef;
           tign[kl] = rindef;
           tdry[kl] = rindef;
           tcum[kl] = 0.0;
           qcum[kl] = 0.0;  }
     } /* end for k */

/*  Make first estimate of drying start times for all components */
/*  These times are usually brief and make little or no difference */
    r =r0 + 0.25 * dr;
    tf = TempF(fi, r);
    ts = tamb;

    if ( tf <= (tpdry + 10.0)) {
       return 1; }                       /* stop' Igniting fire cannot dry fuel' */

    thd = (tpdry - ts) / (tf - ts);
    tx = 0.5 * (ts + tpdry);

    for ( k = 1; k <= number; k++) {       /* do k = 1 , number */
       factor = dendry[k-1]*fmois[k-1];
       conwet = condry[k-1]+4.27e-04*factor;
       for ( l = 0; l <= k; l++) {
         kl = loc(k, l);
         dia = diam[kl];
         HeatExchange (dia, tf, tx, &hf, &hb, conwet, &en);
         dryt = DryTime(en, thd);
         cpwet = cheat[k-1]+fmois[k-1]*ch2o;
         fac = pow2(0.5*dia)/conwet;
         fac = fac * dendry[k-1] * cpwet;
         dryt = fac * dryt;
         tdry[kl] = dryt;
        }
     }

/*   Next , determine which components are alight in spreading fire */
    tsd=tpdry;
    for ( k = 1; k <= number; k++) {        /* do k = 1 , number */
       c = condry[k-1];
       tigk=tpig[k-1];
       for ( l = 0; l <= k; l++) {      /* do l = 0 , k */
          kl = loc(k, l);
          dryt = tdry[kl];
          if (dryt >= dt)
            continue;
          dia = diam[kl];
          ts = 0.5 * (tsd + tigk);
          HeatExchange (dia, tf, ts, &hf, &hb, c, &e);
          tcum[kl] = Max ((tf-ts)*(dt-dryt), 0.0);
          qcum[kl] = hb * tcum[kl];
          if ( tf <= (tigk + 10.0))
             continue;
          dtign = TIgnite (tpdry, tpig[k-1], tf, condry[k-1], cheat[k-1], fmois[k-1], dendry[k-1], hb);
          trt = dryt + dtign;
          tign[kl] = 0.5 * trt;
          if ( dt > trt )
             flit[k-1] += xmat[kl]; } }
    nlit = 0;
    trt = rindef;

/*  Determine minimum ignition time and verify ignition exists */
    for ( k = 1; k <= number; k++) {         /* do k = 1 , number */
      if ( flit[k-1] > 0.0)
        nlit += 1;
      for ( l = 0; l <= k; l++) {            /* do l = 0 , k */
        kl = loc(k, l);
        trt = Min(trt, tign[kl]); } }

   if ( nlit == 0)
     return 2;                               /* stop' START ignites no fuel' */

/*  Deduct trt from all time estimates , resetting time origin */
   for ( k = 1; k <= number; k++) {          /* do k = 1 , number */
      for ( l = 0; l <= k; l++) {            /* do l = 0 , k */
         kl = loc(k, l);
         if ( tdry[kl] < rindef)
            tdry[kl] -= trt;
         if ( tign[kl] < rindef)
            tign[kl] -= trt;  } }

/*  Now go through all component pairs and establish burning rates */
/*  for all the components that are ignited; extrapolate to end time dt */
   for ( k = 1; k <= number; k++) {          /* do k = 1 , number */
     if ( flit[k-1] == 0.0) {
        for ( l = 0; l <= k; l++) {          /* do l = 0 , k */
           kl = loc(k, l);
           ddot[kl] = 0.0;
           tout[kl] = rindef;
           wodot[kl] = 0.0;  }  }
     else  {
        ts = tchar[k-1];
        c = condry[k-1];
        for ( l = 0; l <= k; l++) {    /* do l = 0 , k */
           kl = loc (k, l);
           dia = diam[kl];
           HeatExchange (dia, tf, ts, &hf, &hb, c, &e);
           qdot[kl][now-1] = hb * Max((tf-ts), 0.0);
           aint = pow2 (c/hb);
           ddt = dt - tign[kl];
           acum[kl] = aint * ddt;
           ddot[kl] = qdot[kl][now-1] * work[k-1];
           tout[kl] = dia / ddot[kl];
           dnext = Max(0.0, (dia - ddt * ddot[kl]));
           wnext = WOO[kl] * pow2(dnext / dia);
           wodot[kl] = ( WOO[kl] - wnext) / ddt;
           diam[kl] = dnext;
           *ad_Con += Get_Cons (WOO[kl], wnext);  /* Accum Tot consumd    */

// F-62 completion/acceptance-recovery pass (2026-09-21) DIAGNOSTIC HOOK.
// Opt-in via the SAME FOFEM_TEST_BURN_DIAG env var as the _CompDump()
// hook above (see that function's own header comment for the full
// provenance/diff proof this file carries). Dumps every Start()-computed
// per-fuel-component-pair initial burning-rate quantity to a SEPARATE
// file (never touching _FireInt-Dump.csv's own schema/writer) exactly
// once (Start() itself runs exactly once per BRN_Run invocation, unlike
// the per-timestep _CompDump hook). No equation, arithmetic, or
// simulation state is changed by this block -- it only READS values
// already computed by the two lines directly above.
if ( getenv("FOFEM_TEST_BURN_DIAG") != NULL ) {
  FILE *fhSD = fopen ("_StartDump.csv", "a");
  if ( fhSD != NULL ) {
    static bool wroteHeader = false;
    if ( !wroteHeader ) {
      fprintf (fhSD, "k,l,kl,sigma,dia,hb,c,ddt,tign,work,ddot,WOO_before,WOO_after,wodot\n");
      wroteHeader = true;
    }
    fprintf (fhSD, "%ld,%ld,%ld,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
             k, l, kl, sigma[k-1], dia, hb, c, ddt, tign[kl], work[k-1], ddot[kl], WOO[kl], wnext, wodot[kl]);
    fclose (fhSD);
  }
}

           WOO[kl] = wnext;

           df = 0.0;
           if ( dnext <= 0.0 ) {
              df = xmat[kl];
              if ( kl == 0 )
                 gd_Fudge1 = wodot[kl];      /* Save these for FireIntensity */
              else                           /* function to use              */
                 gd_Fudge2 = wodot[kl];
              wodot[kl] = 0.0;
              ddot[kl] = 0.0;  }

           flit[k-1] -= df;
           fout[k-1] += df;
 
        } /* for l */  
      } /* else */ 
    }  /* for k */ 

   *ncalls = 0;
   return 0;
}


/*----------------------------------------------------------------------------
//
//  Step
//
//----------------------------------------------------------------------------
// subroutine STEP( dt , MXSTEP , now , maxno , number , wo , alfa ,
//   +                 dendry , fmois , cheat , condry , diam , tpig ,
//   +                 tchar , xmat , tambb , tpdry , fi , flit , fout ,
//   +                 tdry , tign , tout , qcum , tcum , acum , qdot ,
//   +                 ddot , wodot , work , u , d , r0 , dr , ch2o ,
//   +                 ncalls , maxkl , tin , fint , fid )
//
//  Updates status of all fuel component pairs and returns a snapshot
//
//   Input parameters:
//
//        tin =           start of current time step
//        dt =            time step , sec
//        MXSTEP =        max dimension of historical sequences
//        now =           index marks end of time step
//        maxno =         max number of fuel components
//        number =        actual number of fuel components
//        wo =            current ovendry loading for the larger of
//                        each component pair, kg / sq m
//        alfa =          dry thermal diffusivity of component , sq m / s
//        dendry =        ovendry density of component , kg / cu m
//        fmois =         moisture fraction of component
//        cheat =         specific heat capacity of component , J / kg K
//        condry =        ovendry thermal conductivity , w / sq m K
//        diam =          current diameter of the larger of each
//                        fuel component pair , m
//        tpig =          ignition temperature ( K ) , by component
//        tchar =         end - pyrolysis temperature ( K ) , by component
//        xmat =          table of influence fractions between components
//        tambb =         ambient temperature ( K )
//        tpdry =         temperature ( all components ) start drying ( K )
//        fi =            current fire intensity ( site avg ) , kW / sq m
//        work( k ) =     factor of heat transfer rate hbar * (Tfire - Tchar)
//                        that yields ddot( k )
//        fint( k ) =     correction to fi to compute local intensity
//                        that may be different due to k burning
//        fid =           fire intensity due to duff burning ... this is
//                        used to up the fire intensity for fuel pieces
//                        that are burning without interacting with others
//        plus the following constants and bookkeeping parameters
//        u , d , r0 , dr , ch20 , ncalls , maxkl
//
//   Parameters updated [input and output]
//
//        ncalls =        counter of calls to this routine ...
//                                = 0 on first call or reset
//                                cumulates after first call
//        flit =          fraction of each component currently alight
//        fout =          c
//        tdry =          time of drying start of the larger of each
//                        fuel component pair
//        tign =          ignition time for the larger of each
//                        fuel component pair
//        tout =          burnout time of larger component of pairs
//        qcum =          cumulative heat input to larger of pair , J / sq m
//        tcum =          cumulative temp integral for qcum ( drying )
//        acum =          heat pulse area for historical rate averaging
//        qdot =          history ( post ignite ) of heat transfer rate
//                        to the larger of component pair , W / sq m
//        ddot =          diameter reduction rate , larger of pair , m / s
//        wodot =         dry loading loss rate for larger of pair
//
//   Constant parameters
//
//        u =             mean horizontal windspeed at top of fuelbed
//        d =             fuelbed depth
//        r0 =            minimum value of mixing parameter
//        dr =            max - min value of mixing parameter
//        ch2o =          specific heat capacity of water , J / kg K
//        hvap =          heat of vaporization of water , J / kg
*/
/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
void Step(double dt, double tin, double fid, long *ncalls, double *ad_Con)
{
int nspan;
long k, l, kl, now, mu, index;
double c, rindef=1.e+30;
double tnext, tnow, tgo, tdun, tifi;
double aint, tav1, tav2, tav3, tavg;
double qdavg, qdsum, tspan, deltim;
double tlit, ts, r, gi, tf, dia, hf, hb;
double e, qqq, tst, dnext, wnext, rate, ddt, dryt, dtemp, dqdt;
double dteff, heff, tfe, dtlite, qd, delt, factor;
double conwet, dtcum, he, dtef, thd, biot, cpwet, fac;
bool flag;

   *ad_Con = 0;
   *ncalls += 1;
    now = *ncalls;
    tnow = tin;
    tnext = tnow + dt;
/*  tifi = time when fire ignition phase ended ( at now = 1 ) */
    tifi = tnow-((double)(now-1))*dt;

    for (k = 1; k <= number; k++) {
       c = condry[k-1];
       for ( l = 0; l <= k; l++) {
         kl = loc(k, l);
         tdun = tout[kl];
         if ( tnow >= tdun) {      /* See if k of ( k , l ) pair burned out */
           ddot[kl] = 0.0;
           wodot[kl] = 0.0;
           continue; }                  /*   goto 10 */

         if ( tnext >= tdun ) {
           tgo = tdun - tnow;
           ddot[kl] = diam[kl] / tgo;
           wodot[kl] = WOO[kl] / tgo;
           *ad_Con += Get_Cons (WOO[kl],0.0);
           WOO[kl] = 0.0;
           diam[kl] = 0.0;
           continue; }

/*  k has not yet burned out ... see if k of ( k , l ) pair is ignited */
         tlit = tign[kl];
         if ( tnow >= tlit ) {
            ts = tchar[k-1];
            if ( l == 0 ) {
              r = r0 + 0.5 * dr;
              gi = fi + fid; }
            else if ( l == k ) {
              r = r0 + 0.5 * (1.0 + flit[k-1]) * dr;
              gi = fi + flit[k-1] * fint[k-1]; }
            else {                                       /* (l!=0 && l!=k) */
              r = r0 + 0.5 * (1.0 + flit[l-1]) * dr;
              gi = fi + fint[k-1] + flit[l-1] * fint[l-1];  }
            tf = TempF(gi, r);
            dia = diam[kl];
            HeatExchange (dia, tf, ts, &hf, &hb, c, &e);
            qqq = hb * Max(tf-ts, 0.0);
            tst = Max(tlit, tifi);
            nspan = Max(1, Nint((tnext-tst)/dt));        /* nint((tnext-tst)/dt)); */
            if ( nspan <= MXSTEP)
              qdot[kl][nspan-1]=qqq;
            else if ( nspan > MXSTEP) {
               for ( mu = 2; mu <= MXSTEP; mu++)     /* do mu = 2 , MXSTEP */
                 qdot[kl][mu-2]=qdot[kl][mu-1];
               qdot[kl][MXSTEP-1]=qqq; }
               aint = pow2(c/hb);
               acum[kl] += (aint * dt);
               tav1 = tnext-tlit;
               tav2 = acum[kl] / alfa[k-1];
               tav3 = pow2(dia/4.0) / alfa[k-1];
               tavg = tav1;
               if (tav2 < tavg)
                 tavg = tav2;
               if ( tav3 < tavg)
                 tavg = tav3;
               index = Min(nspan, MXSTEP); /* index = 1+min(nspan, MXSTEP); */
               qdsum = 0.0;
               tspan = 0.0;
               deltim = dt;
               do {
                 index-=1;
                 if ( index == 0 )                          /* ==1 */
                   deltim =tnext-tspan-tlit;
                 if (( tspan + deltim) >= tavg)
                   deltim = tavg - tspan;
                 qdsum += (qdot[kl][index] * deltim);
                 tspan += deltim;
                 if (tspan >= tavg)
                   break;
               } while(index>0);

              qdavg = Max(qdsum / tspan, 0.0);
              ddot[kl] = qdavg * work[k-1];
              dnext = Max(0.0, dia-dt * ddot[kl]);
              wnext = WOO[kl] * pow2(dnext/dia);
              if (( dnext == 0.0) && ( ddot[kl] > 0.0))
                tout[kl] = tnow + dia / ddot[kl];
              else if (( dnext > 0.0) && ( dnext < dia)) {
                rate = dia / ( dia - dnext);
                tout[kl] = tnow + rate * dt; }
              if ( qdavg <= (double) MXSTEP)       /* <=20.0 in Albini's code */
                tout[kl] = 0.5 * (tnow+tnext);
              ddt = Min(dt, (tout[kl]-tnow));
              wodot[kl] = (WOO[kl] - wnext) / ddt;
              diam[kl] = dnext;
              *ad_Con += Get_Cons (WOO[kl],wnext);
              WOO[kl] = wnext;
              continue; }

 /*   See if k of ( k , l ) has reached outer surface drying stage yet */
            dryt = tdry[kl];
            if ( tnow >= dryt && tnow<tlit) {
              if ( l == 0 ) {
                r = r0;
                gi = fi + fid; }
              else if ( l == k ) {
                r = r0;
                gi = fi; }
            else {                                    /*  if(l!=0 && l!=k) */
              r = r0+0.5 * flit[l-1] * dr;
              gi = fi + flit[l-1] * fint[l-1]; }
            tf = TempF(gi, r);
            ts = tamb;
            dia = diam[kl];
            HeatExchange(dia, tf, ts, &hf, &hb, c, &e);
                  /* call heatx( u , d , dia , tf , ts , hf , hb , c , e ) */
            dtemp = Max(0.0, tf-ts);
            dqdt = hb * dtemp;
            qcum[kl] += (dqdt * dt);
            tcum[kl] += (dtemp * dt);
            dteff = tcum[kl] / (tnext-dryt);
            heff = qcum[kl] / tcum[kl];
            tfe = ts + dteff;
            dtlite = rindef;
            if ( tfe > (tpig[k-1] + 10.0))
              dtlite = TIgnite(tpdry,tpig[k-1],tfe,condry[k-1],cheat[k-1],fmois[k-1], dendry[k-1], heff);
            tign[kl] = 0.5 * (dryt + dtlite);

/* If k will ignite before time step over , must interpolate */
            if ( tnext > tign[kl]){
              ts = tchar[k-1];
              HeatExchange(dia, tf, ts, &hf, &hb, c, &e);
              qdot[kl][0] = hb * Max(tf-ts, 0.0);
              qd = qdot[kl][0];
              ddot[kl] = qd * work[k-1];
              delt = tnext - tign[kl];
              dnext = Max(0.0, dia-delt*ddot[kl]);
              wnext = WOO[kl] * pow2(dnext/dia);
              if ( dnext == 0.0)
                tout[kl] = tnow+dia / ddot[kl];
              else if ((dnext>0.0) && dnext < dia) {
                rate = dia / (dia-dnext);
                tout[kl] = tnow + rate * dt;  }
              if ( tout[kl] > tnow) {
                ddt = Min(dt, (tout[kl]-tnow));
                wodot[kl] = (WOO[kl] - wnext) / ddt; }
              else
                wodot[kl] = 0.0;

              diam[kl] = dnext;
              *ad_Con += Get_Cons (WOO[kl],wnext);
              WOO[kl] = wnext; }
            continue;
        } /* end if */

 /*   If k of ( k , l ) still coming up to drying temperature , accumulate */
 /*   heat input and driving temperature difference , predict drying start */

        if ( tnow < dryt) {
           factor = fmois[k-1] * dendry[k-1];
           conwet = condry[k-1]+4.27e-04*factor;
           if ( l == 0 ) {
             r = r0;
             gi = fi + fid;  }
           else if ( l == k) {
             r = r0;
             gi = fi;  }
           else if (( l != 0) && ( l != k )) {
             r = r0+0.5 * flit[l-1] * dr;
             gi = fi + flit[l-1] * fint[l-1];  }

           tf = TempF(gi, r);
           if ( tf <= (tpdry + 10.0))
             continue;       /*  goto 10 */
           dia = diam[kl];
           ts = 0.5 * (tamb + tpdry);
           HeatExchange(dia, tf, ts, &hf, &hb, c, &e);
           dtcum = Max((tf-ts)*dt, 0.0);
           tcum[kl] += dtcum;
           qcum[kl] += ( hb * dtcum);
           he = qcum[kl] / tcum[kl];
           dtef = tcum[kl] / tnext;
           thd = (tpdry - tamb) / dtef;
           if ( thd > 0.9)
              continue;
           biot = he * dia / conwet;
           dryt = DryTime(biot, thd);
           cpwet = cheat[k-1] + ch2o * fmois[k-1];
           fac = pow2(0.5 * dia) / conwet;
           fac = fac * cpwet * dendry[k-1];
           tdry[kl] = fac * dryt;
           if (tdry[kl] < tnext)  {
              ts = tpdry;
              HeatExchange(dia, tf, ts, &hf, &hb, c, &e);
              dqdt = hb * (tf-ts);
              delt = tnext-tdry[kl];
              qcum[kl] = dqdt * delt;
              tcum[kl] = (tf-ts) * delt;

/* tbar not used anywhere  tbar=0.5*(tpdry+tpig[k-1]); */
/* See if ignition to occur before time step complete */
              if ( tf <= (tpig[k-1] + 10.0))
                 continue;
              dtlite = TIgnite (tpdry,tpig[k-1],tf,condry[k-1],cheat[k-1], fmois[k-1], dendry[k-1], hb);
              tign[kl] = 0.5 * (tdry[kl] + dtlite);
             if ( tnext > tign[kl]) {
               ts = tchar[k-1];
               qdot[kl][0] = hb * Max(tf-ts, 0.0); } } }
     }  /* end for l */
   }    /* end for k */


/*  Update fractions ignited and burned out , to apply at next step start */
   for ( k = 1; k <= number; k++) {
     flit[k-1] = 0.0;
     fout[k-1] = 0.0;
     for ( l = 0; l <= k; l++ ) {
       kl = loc(k, l);
       if ( tnext >= tign[kl])
          flag = true;
       else
          flag = false;
       if ( flag && tnext <= tout[kl])
          flit[k-1] += xmat[kl];
       if (tnext > tout[kl])
          fout[k-1] += xmat[kl]; } }
}
/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Summary
* Desc: Creates the output
*       This is where I get Remaining Load amounts
*   In: cr_LoadFN...file name, if null string then no file is created
*  Ret: 1 OK
*       0 Error opening output file
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/

double KgSq_To_Pnd (double d);

int Summary (double d_Time, char cr_LoadFN[],char cr_ErrMes[])
{
int iN; 
long m, n, mn;
double d, ti, ts, to, tf, wd, rem;
FILE *outfile;  

   iN = 0; 
   if ( strcmp ( cr_LoadFN,"" )) {
      outfile = fopen(cr_LoadFN, "w");
      if ( outfile == NULL ) {
        sprintf (cr_ErrMes, "Can't open Burnup output consumed file.\n%s",cr_LoadFN);
        return 0; } }

   if ( strcmp ( cr_LoadFN,"" )) {
     fprintf(outfile, "Column 1: Fuel component\n");
     fprintf(outfile, "       2: Preburn  fuel load (%s)\n", e_KgSqM);
     fprintf(outfile, "       3: Postburn fuel load (%s)\n", e_KgSqM); 

     fprintf(outfile, "       4: Time until ignition (%s)\n", e_Second);
     fprintf(outfile, "       5: Time until burnout  (%s) \n", e_Second);

     fprintf(outfile, "       6: Moisture content  (0->1.0)\n");
     fprintf(outfile, "       7: Sigma, surface area to volume ratio (per m)\n");

     fprintf(outfile, "       8: Preburn  fuel load (%s) \n", e_TonAc);
     fprintf(outfile, "       9: Postburn fuel load (%s) \n",  e_TonAc); 

     fprintf(outfile, "      10: Preburn  fuel load (%s) \n", e_lbAc);
     fprintf(outfile, "      11: Postburn fuel load (%s) \n", e_lbAc); 
    
     fprintf(outfile, "\n       %6.0f  Duration of fire (%s)\n\n",d_Time, e_Second); 
   }

/*-----------------------------------------------------------*/
   for( m = 1; m <= number; m++) {         /* Each Component */
  /*   if ( strcmp ( cr_LoadFN,"" ))  */
  /*      fprintf(outfile, "%5ld %8.5lf (%8.5lf) %6.2lf %8.2lf", m, wdry[m-1], KgSq_To_TPA(wdry[m-1]), fmois[m-1], 4.0/sigma[m-1]);  */ 
    
      rem = 0.0;
      ts = 1.0e31;
      tf = 0.0;
      d = 999999999999;

      for ( n = 0; n <= m; n++) {
         mn = loc(m, n);

         ti = tign[mn];
         ts = Min(ti, ts);

         to = tout[mn];
         tf = Max(to, tf);

         wd = WOO[mn];
         rem = rem + wd; 

      } /* for n */

      if ( strcmp ( cr_LoadFN,"" ))
/*      fprintf(outfile, " %15.0lf %15.0lf %11.5lf (%8.5lf)     %6.0f\n", ts, d, rem, KgSq_To_TPA(rem), dr_Done[m-1]); */
        fprintf(outfile, "%5ld   %8.5lf %8.5lf    %6.0f --> %6.0f   %6.2lf %8.2lf   %8.5lf %8.5lf    %6.0f %6.0f\n", 
                   m, 
                   wdry[m-1],   rem,                                /* Load Kg */
                   ts,  dr_Done[m-1],                             /* Times */
                   fmois[m-1],
/* change 4-16-2015original                   4.0/sigma[m-1], */
                   sigma[m-1],
                   KgSq_To_TPA(wdry[m-1]), KgSq_To_TPA (rem), 
                   KgSq_To_Pnd(wdry[m-1]), KgSq_To_Pnd (rem) ); 
/*    fmois[m-1], 4.0/sigma[m-1]); */
    
      BOV_PutRemAmt (rem, m, tf);            /* Send out Remaining Load Amounts   */

   }  /* for m */

  if ( iN > 0 ) {
     fprintf (outfile, "\n\n  -1.0 indicates the fuel component didn't ignite or that it was"); 
     fprintf (outfile, " burning at such an insignificant the fire was considered out.");   }     

  if ( strcmp ( cr_LoadFN,"" )) 
    fclose(outfile);

  return 1;
 }

/*******************************************************
*
********************************************************/
double KgSq_To_Pnd (double d)
{
double dd;
  dd = KgSq_To_TPA(d);
  dd = dd * 2000.0; 
  return dd; 
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Summary
* Desc: Creates the output
*       This is where I get Remaining Load amounts
*   In: cr_LoadFN...file name, if null string then no file is created
*  Ret: 1 OK
*       0 Error opening output file
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int Summary_Old (char cr_LoadFN[],char cr_ErrMes[])
{
int iN; 
long m, n, mn;
double d, ti, ts, to, tf, wd, rem;
FILE *outfile;

   iN = 0; 
   if ( strcmp ( cr_LoadFN,"" )) {
      outfile = fopen(cr_LoadFN, "w");
      if ( outfile == NULL ) {
        strcpy (cr_ErrMes, "Can't open Burnup output consumed file.");
        return 0; } }

   if ( strcmp ( cr_LoadFN,"" )) {
     fprintf(outfile, "Column 1: Fuel component\n");
     fprintf(outfile, "       2: Preburn fuel load, (%s)\n", e_KgSqM);
     fprintf(outfile, "       3: Moisture content, fraction\n");
     fprintf(outfile, "       4: Sigma, surface area to volume ratio, (%s)\n", e_SAV);
     fprintf(outfile, "       5: Time utill ignition, (%)\n", e_Second);
     fprintf(outfile, "       6: Time utill burnout, (%s)\n", e_Second);
     fprintf(outfile, "       7: Postburn fuel load, (%s)\n\n", e_KgSqM); }

   for( m = 1; m <= number; m++) {
     if ( strcmp ( cr_LoadFN,"" ))
 /*      fprintf(outfile, "%5ld %8.2lf %6.2lf %8.2lf", m, wdry[m-1], fmois[m-1], 4.0/sigma[m-1]); */
       fprintf(outfile, "%5ld %8.5lf (%8.5lf) %6.2lf %8.2lf", m, wdry[m-1], KgSq_To_TPA(wdry[m-1]), fmois[m-1], 4.0/sigma[m-1]);
    
     rem = 0.0;
     ts = 1.0e31;
     tf = 0.0;
     d = 999999999999;
     for ( n = 0; n <= m; n++) {
       mn = loc(m, n);
       ti = tign[mn];
       ts = Min(ti, ts);
       to = tout[mn];
       tf = Max(to, tf);
       wd = WOO[mn];
       rem = rem + wd; 

       d = Max (ti,d);

     } /* for n */

    if ( strcmp ( cr_LoadFN,"" )) {
      if ( tf > 100000000.0 ) { 
         d = -1.0;
         iN++; }
      else 
         d = tf; 
/*      fprintf(outfile, " %15.5lf %15.5lf %11.5lf\n", ts, d, rem); */
      fprintf(outfile, " %15.0lf %15.0lf %11.5lf (%8.5lf)     %6.0f\n", ts, d, rem, KgSq_To_TPA(rem), dr_Done[m-1]);
    }  /* if */

    BOV_PutRemAmt (rem, m, tf);            /* Send out Remaining Load Amounts   */

  }  /* for m */

  if ( iN > 0 ) {
     fprintf (outfile, "\n\n  -1.0 indicates the fuel component didn't ignite or that it was"); 
     fprintf (outfile, " burning at such an insignificant the fire was considered out.");  
   }     

  if ( strcmp ( cr_LoadFN,"" )) 
    fclose(outfile);

  return 1;
 }


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: BRN_Init
* Desc: Init all variables need to run Burnup
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
void BRN_Init ()
{
long i;

   ntimes = 0;
   number = 0;
   fi = 0.0;
   ti = 0.0;
   u  = 0.0;
   d  = 0.0;
   tamb = 0.0;
   ak = 0.0;
   r0 = 0.0;
   dr = 0.0;
   dt = 0.0;
   wdf  = 0.0;
   dfm  = 2.0;

   for( i = 0; i < MAXNO; i++) {
      wdry[i]   = 0.0;
      ash[i]    = 0.0;
      htval[i]  = 0.0;
      fmois[i]  = 0.0;
      dendry[i] = 0.0;
      sigma[i]  = 0.0;
      cheat[i]  = 0.0;
      condry[i] = 0.0;
      alfa[i]   = 0.0;
      tpig[i]   = 0.0;
      tchar[i]  = 0.0;
      flit[i]   = 0.0;
      fout[i]   = 0.0;
      work[i]   = 0.0;
      alone[i]  = 0.0;
      area[i]   = 0.0;
      fint[i]   = 0.0; }
 }


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: CheckData
* Desc: Check inputs before running burnup
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int  BRN_CheckData (char cr_ErrMes[])
{
long i;

const double tch1 = e_tch1 + e_tchar_adj;
const double tch2 = e_tch2 + e_tchar_adj;

const double tig1 = e_tig1 + e_tpig_adj;
const double tig2 = e_tig2 + e_tpig_adj;

const double cht1 = e_cht1,  cht2 = e_cht2;

const double ash1 = 0.0001, ash2 = 0.1;
const double htv1 = 1.0e07, htv2 = 3.0e7;
const double den1 = 200.0,  den2 = 1000.0;
const double sig1 = 4.0,    sig2 = 1.0e4;
const double con1 = 0.025,  con2 = 0.25;
const double fir1 = 40.0,   fir2 = 1.0e5;
const double ti1  = 10.0,   ti2  = 200.0;
const double u1   = 0.0,    u2   = 5.0;
const double d1   = 0.1,    d2   = 5.0;
const double tam1 = -40.0,  tam2 = 40.0;

char cr[200];
   strcpy (cr_ErrMes,"");

   if ( ntimes <= 0 ) {
     sprintf(cr_ErrMes, "Burnup: Number of iterations invalid, %d, must be > 0",ntimes);
     return false; }

   for (i = 0; i < number; i++) {

     if ( wdry[i] <= e_small || wdry[i] >= e_big) {
       sprintf (cr_ErrMes, "Burnup: dry loading out of range (%s): %8.2f, Low:%8.2f Up:%8.2f",e_KgSqM, wdry[i],e_small, e_big);
       return false; }

     if ( ash[i]<=ash1 || ash[i]>=ash2) {
       sprintf (cr_ErrMes, "Burnup: ash content out of range (fraction): %8.2f, Low:%8.2f Up:%8.2f", ash[i], ash1, ash2);
       return false; }

     if ( htval[i] <= htv1 || htval[i] >= htv2) {
       sprintf (cr_ErrMes, "Burnup: heat content out of range (%s): %8.2f, Low:%8.2f Up:%8.2f", e_JoulKg, htval[i], htv1, htv2);
       return false;}

     if ( !_ChkLimM(fmois[i], e_fms1, e_fms2) ) {
       sprintf (cr_ErrMes, "Burnup: fuel moisture out of range (fraction): %6.2f, (Limits:%6.2f ->%6.2f)",fmois[i],e_fms1,e_fms2);
       return false; }

     if ( dendry[i]<=den1 || dendry[i]>=den2)  {
       sprintf (cr_ErrMes, "Burnup: dry mass density out of range (%s): %8.2f, Low:%8.2f Up:%8.2f", e_KgCuM, dendry[i],den1,den2);
       return false; }

     if ( sigma[i]<=sig1 || sigma[i]>=sig2) {
       sprintf (cr_ErrMes, "Burnup: sigma out of range (%s): %8.2f, Low:%8.2f Up:%8.2f", e_SAV, sigma[i],sig1,sig2);
       return false; }

     if ( cheat[i]<=cht1 || cheat[i]>=cht2) {
       sprintf (cr_ErrMes, "Burnup: heat capacity out of range (J/kg/K: %8.2f, Low:%8.2f Up:%8.2f",cheat[i], cht1, cht2);
       return false; }

     if ( condry[i]<=con1 || condry[i]>=con2) {
       sprintf (cr_ErrMes, "Burnup: thermal conductivity out of range (W/m/K): %8.2f, Low:%8.2f Up:%8.2f",condry[i],con1,con2);
       return false; }


     if ( tpig[i]<=tig1 || tpig[i]>=tig2) {
       sprintf (cr_ErrMes, "Burnup: ignition temperature out of range (%s): %8.2f\n Limits:   Lower:%8.2f    Upper:%8.2f", e_Celsius, 
                tpig[i]-e_tpig_adj,   tig1-e_tpig_adj,       tig2-e_tpig_adj);
       return false; }


     if ( tchar[i] <=tch1 || tchar[i]>=tch2) {
       sprintf (cr_ErrMes, "Burnup: char end pyrolisis temperature out of range (%s): %8.2f\n Limits:   Lower:%8.2f   Upper:%8.2f", e_Celsius, 
                  tchar[i]-e_tchar_adj,  tch1-e_tchar_adj,  tch2-e_tchar_adj);
       return false; }

   }  /* for i */

   if ( fi < fir1 || fi > fir2)
      sprintf (cr_ErrMes, "Burnup: igniting fire intensity out of range (%s)", e_KwSqM);

   else if ( ti < ti1 || ti > ti2 )
      sprintf (cr_ErrMes, "Burnup: igniting surface fire res. time out of range (%s)", e_Second);

   else if ( u < u1 || u > u2)
      sprintf (cr_ErrMes, " windspeed at top of fuelbed out of range (%s)", e_MetSec);

   else if ( d < d1 || d > d2) {
      sprintf (cr_ErrMes, "Burnup: depth of fuel bed out of range (%s)\n", e_Meter);
      sprintf ( cr, "Depth: %8.2f,   Limit Low:%8.2f,  High:%8.2f\n", d, d1, d2);
      strcat (cr_ErrMes, cr); }

   else if ( tamb-273 < tam1 || tamb-273 > tam2)
      sprintf (cr_ErrMes, "Burnup: ambient temperature out of range (%s)", e_Celsius);

   else if ( ( wdf < e_wdf1 || wdf > e_wdf2) &&  ( wdf != 0 ) )
      sprintf(cr_ErrMes, "Burnup: duff dry weight loading, %6.2f is out of range (%s), Limit: %6.2f -> %6.2f", wdf, e_KgSqM, e_wdf1,e_wdf2);

   else if ( dfm < e_dfm1 || dfm > e_dfm2)
      strcpy(cr_ErrMes, "Burnup: duff moisture out of range (fraction)");

   if ( strlen(cr_ErrMes) > 0)
     return false;

   return true;
 }

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: _ChkLimM
* Desc: Check a value against lower and upper limits.
* NOTE: I was having a problem with floating point percission with
*        the lower moisture limit, 0.01 was being stored as 0.0099999999...
*        causing an error when comparing the lower limit
*   In: d value to compare, low and hi limits
*  Ret: 1 ok, 0 out of limits
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int _ChkLimM (double d, double low, double hi)
{
int D,LOW,HI;
float f;
   f = (d * 100.0);                     /* Had to do it like this, It was    */
   D = (int) f;                         /*  losing the decimal digits        */

   f = (low * 100.0);
   LOW = (int) f;

   f = (hi * 100.0);
   HI = (int) f;

   if ( D < LOW  || D > HI )
     return 0;

   return 1;

}



/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: BRN_SetFireDat
* Desc: Set some of the Fire Input data that are need in place before running
*       burnup.
*       The number of iterations is checked here
* Note: Also see  BRN_CheckData()
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
void BRN_SetFireDat(long NumIter, double Fi, double Ti, double U,  double D,
                    double Tamb,  double R0, double Dr, double Dt, double Wdf,
                    double Dfm)
{
   ntimes = NumIter;
   fi=Fi;
   ti=Ti;
   u=U;
   d=D;
   tamb=Tamb+273.0;
   r0=R0;
   dr=Dr;
   dt=Dt;
   wdf=Wdf;
   dfm=Dfm;
 }

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name:
* Desc:
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
 long loc(long k, long l)
{
  return k*(k+1.0)/2.0+l-1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name:
* Desc:
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
long Nint(double input)
{
long Input;
    Input = (long) input;
    if (input - (double)Input  >=  0.5)
       Input += 1;
    return Input;
}


/*--------------------------------------------------------------------------*/
/*                                                                          */
/*      Tignit                                                              */
/*                                                                          */
/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
double ff(double x, double tpfi, double tpig)
{
const double a03=-1.3371565;
const double a13=0.4653628;
const double a23=-0.1282064;
double b03;

   b03 = a03 * (tpfi - tpig) / (tpfi - tamb);
   return b03+x*(a13+x*(a23+x));
}


/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/*   subroutine TIGNIT( tpam , tpdr , tpig , tpfi , cond ,
//    +                   chtd , fmof , dend , hbar , tmig )
//c   tpam = ambient temperature , K
//c   tpdr = fuel temperature at start of drying , K
//c   tpig = fuel surface temperature at ignition , K
//c   tpfi = fire environment temperature , K
//c   cond = fuel ovendry thermal conductivity , W / m K
//c   chtd = fuel ovendry specific heat capacity , J / kg K
//c   fmof = fuel moisture content , fraction dry weight
//c   dend = fuel ovendry density , kg / cu m
//c   hbar = effective film heat transfer coefficient [< HEATX] W / sq m K
//c   tmig = predicted time to piloted ignition , s
*/
/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/

double TIgnite(double tpdr, double tpig, double tpfi,
              double cond, double chtd, double fmof, double dend, double hbar)
{
const double pinv = 2.125534;
const double hvap = 2.177e+06;
const double cpm = 4186.0;
const double conc = 4.27e-04;
double xlo, xhi, xav, fav, beta, conw, dtb, dti, ratio, rhoc, tmig;

/* MOVED TO function FF(...)
//------------------------------------------------------------------------------
//c   approximate function of beta to be solved is ff( x ) where
//c   x  =  1 / ( 1 + p * beta )  { Hastings, Approximations for
//c   digital computers } and we employ      pinv  =  1 / p
//      ff( x ) = b03 + x * ( a13 + x * ( a23 + x ) )
//c   radiant heating equivalent form gives end condition fixes beta
//      b03 = a03 * ( tpfi - tpig ) / ( tpfi - tpam )
//------------------------------------------------------------------------------
//c   find x that solves ff( x ) = 0 ;  method is binary search
*/
      xlo = 0.0;
      xhi = 1.0;

      do {
        xav = 0.5 * ( xlo + xhi );
        fav = ff (xav, tpfi, tpig);
        if ( fabs(fav) > e_small ) {
          if ( fav < 0.0)
            xlo = xav;
          if ( fav > 0.0)
            xhi = xav;  }
      } while ( fabs(fav) > e_small );

      beta = pinv*(1.0-xav)/xav;
      conw = cond+conc*dend*fmof;
      dtb        = tpdr-tamb;
      dti        = tpig-tamb;
      ratio = (hvap+cpm*dtb)/(chtd*dti);
      rhoc = dend*chtd*(1.0+fmof*ratio);
      tmig = pow2(beta/hbar)*conw*rhoc;

      return tmig;
 }


/*------------------------------------------------------------------------------
//
//      Arrays
//
//-----------------------------------------------------------------------------*/
void Arrays()
{
/*  subroutine ARRAYS( maxno , number , wdry , ash , dendry , fmois ,
//    +                   sigma , htval , cheat , condry , tpig , tchar ,
//    +                   diam , key , work , ak , elam , alone , xmat ,
//    +                   wo , maxkl )
//  Orders the fuel description arrays according to the paradigm described in
//  subroutine SORTER and computes the interaction matrix xmat from the array
//  elam and the list alone returned from subroutine OVLAPS.  Parameters in
//  arrays are defined in terms of initial values as:
//       wdry            ovendry mass loading , kg / sq m
//       ash             mineral content , fraction dry mass
//       dendry          ovendry mass density , kg / cu m
//       fmois           moisture content , fraction dry mass
//       sigma           surface to volume ratio , 1 / m
//       htval           low heat of combustion , J / kg
//       cheat           specific heat capacity , ( J / K ) / kg dry mass
//       condry          thermal conductivity , W / m  K , ovendry
//       tpig            ignition temperature , K
//       tchar           char temperature , K
//       diam            initial diameter , m [ by interaction pairs ]
//       key             ordered index list
//       work            workspace array
//       elam            interaction matrix from OVLAPS
//       alone           noninteraction fraction list from OVLAPS
//       xmat            consolidated interaction matrix
//       wo              initial dry loading by interaction pairs */

double diak, wtk;
long j, k, kl, kj;

    Sorter();       /*  call SORTER( maxno , number , sigma , fmois , dendry , key ) */

    for ( j = 0; j < number; j++) {
      k = key[j];
      work[j]= wdry[k]; }
    for ( j = 0; j < number; j++)
      wdry[j] = work[j];

    for ( j = 0; j < number; j++) {
      k = key[j];
      work[j]= ash[k];  }
    for ( j = 0; j < number; j++)
      ash[j] = work[j];

    for ( j = 0; j < number; j++) {
      k = key[j];
      work[j]= htval[k];  }
    for ( j = 0; j < number; j++)
      htval[j]= work[j];

    for ( j = 0; j < number; j++) {
      k = key[j];
      work[j]= cheat[k];  }
    for ( j = 0; j < number; j++)
      cheat[j] = work[j];

    for ( j = 0; j < number; j++) {
      k = key[j];
      work[j]= condry[k]; }
    for ( j = 0; j < number; j++)
      condry[j] = work[j];

    for ( j = 0; j < number; j++) {
      k = key[j];
      work[j]= tpig[k];       }
    for ( j = 0; j < number; j++)
      tpig[j] = work[j];

    for ( j = 0; j < number; j++) {
      k = key[j];
      work[j]= tchar[k];  }
    for ( j = 0; j < number; j++)
      tchar[j] = work[j];

    OverLaps();   /*   call OVLAPS( wdry , sigma , dendry , ak , number , maxno , maxkl , xmat , elam , alone ) */

    for ( k = 1; k <= number; k++) {             /* do k = 1 , number */
      diak = 4.0 / sigma[k-1];
      wtk = wdry[k-1];
      kl = loc(k, 0);
      diam[kl] = diak;
      xmat[kl] = alone[k-1];
      WOO[kl] = wtk * xmat[kl];
      for ( j = 1; j <= k; j++) {               /* do j = 1 , k */
        kj = loc(k, j);
        diam[kj] = diak;
        xmat[kj] = elam[k-1][j-1];
        WOO[kj] = wtk * xmat[kj]; } }
}


/*------------------------------------------------------------------------------
//
//      TempF
//    function TEMPF( q , r , tamb )
//c  Returns a fire environment temperature , TEMPF , given the fire intensity
//c  q in kW / square meter , the ambient temperature tamb in Kelvins, and the
//c  dimensionless mixing parameter r.
//
//--------------------------------------------------------------------------*/
 double TempF(double q, double r)
{
const double err=1.0e-04;
const double aa=20.0;
double term, rlast, den, rnext, test, tempf;

/* NOTE: I put a check in because this would do a divide by 0,              */
   if ( q != 0 )
     term = r / (aa * q);
   else
     term = 0;

   rlast = r;
   do {
     den = 1.0 + term*(rlast+1.0)*(rlast*rlast+1.0);
     rnext = 0.5*(rlast+1.0+r/den);
     test = fabs(rnext-rlast);
     if ( test < err) {
        tempf = rnext * tamb;
        break;  }
     rlast = rnext;
   } while ( test >= err);

   return tempf;
}

/*------------------------------------------------------------------------------
//
//      Sorter
//
//------------------------------------------------------------------------------
//      subroutine SORTER( maxno , number , sigma , fmois , dendry , key )
//
//  Sorts fuel element list in order of increasing size (decreasing sigma)
//  For elements with same size, order determined on increasing moisture
//  content (fmois).  If items have same size and moisture content, order
//  on the basis of increasing mass density (dendry).  "number" elements are
//  included in the list, which has a maximum length of "maxno".  The integer
//  list:  key( j ) , j = 1 , number holds the indices in order, so other
//  fuel parameters can be ordered and associated as necessary.   */
void Sorter()
{
 long i, j, keep;
 double s, fm, de, usi;
 bool diam, mois, dens, tied, earlyout;

      for (j = 0; j < MAXNO; j++)     /* do j=1 , maxno */
         key[j]=j;

 /*  Replacement sort: order on increasing size , moisture , density */

      for(j=2; j<=number; j++) {          /* do j = 2 , number */
         s = 1.0 / sigma[j-1];
         fm = fmois[j-1];
         de = dendry[j-1];
         keep = key[j-1];
         for(i=(j-2); i>=0; i--) {      /* do i = ( j - 1 ) , 1 , -1 */
           earlyout=true;
           usi = 1.0/sigma[i];
           if ( usi < s )
              diam=true;
           else
              diam = false;
           if ( diam )
              break;                         /*  goto 10 */
           if ( usi == s )
              tied = true;
           else
              tied = false;
           if ( tied ) {                       /* goto 05 */
              if ( fmois[i] < fm)
                 mois=true;
              else
                 mois = false;
              if( mois )
                 break;  /* goto 10 */
              if(fmois[i]==fm)
                tied=true;
              else
                tied=false;
              if ( tied ) {
                 if(dendry[i]<=de)
                    dens=true;
                 else
                    dens=false;
                 if(dens)
                   break;   /* goto 10 */
              }
            }
                 sigma[i+1]=sigma[i];
                 fmois[i+1]=fmois[i];
                 dendry[i+1]=dendry[i];
                 key[i+1]=key[i];
                earlyout=false;
                 }
           if(!earlyout)
                         i=0;
         sigma[i+1]=1.0/s;
                 fmois[i+1]=fm;
                 dendry[i+1]=de;
                 key[i+1]=keep;
         }
 }

/*------------------------------------------------------------------------------
//
//      OvLaps
//
// wdry , sigma , dendry , ak , number , maxno , maxkl , beta , elam , alone , area )
//  Computes the interaction matrix elam( j , k ) which apportions the
//  influence of smaller and equal size pieces on each size class for the
//  purpose of establishing the rates at which the elements burn out.
//  Input quantities are:  wdry , the ovendry mass per unit area of each
//  element available for burning after the passage of the igniting surface
//  fire;  sigma , the particle's surface / volume ratio , and dendry , the
//  ovendry mass density of the particle; ak a dimensionless parameter that
//  scales the planform area of a particle to its area of influence. There
//  are "number" separate particle classes, of a maximum number = maxno.
//  It is assumed that the lists are ordered on size class (nonincreasing
//  surface / volume ratio). List "alone" gives the fraction of each loading
//  that is not influenced by any other category.
//  double wdry[MAXNO] , sigma[MAXNO] , dendry[MAXNO];
//  double beta[MAXKL] , elam[MAXNO][MAXNO] , alone[MAXNO];
//  double area[MAXNO];
//------------------------------------------------------------------------- */
void OverLaps()
{
long j, k, l, kj, kl;
double a, bb, pi, siga, frac;
      frac = 0;    /* put this in 12-18-00, it wasn't being init */
      pi = fabs(acos(-1.0));
      for(j=1; j<=number; j++)   /* do j = 1 , number */
      {  alone[j-1]=0.0;
           for(k=1; k<=j; k++)   /* do k = 1 , j */
           {     kj = loc(j, k);
                xmat[kj]=0.0;
           }
                 for(k=1; k<=number; k++)   /*  do k = 1 , number */
                 elam[j-1][k-1] = 0.0;
         }
      for(k=1; k<=number; k++)   /* do k = 1 , number */
      {    for(l=1; l<=k; l++)   /* do l = 1 , k */
           {     ak=3.25*exp(-20.0*pow2(fmois[l-1]));
                         siga = ak * sigma[k-1]/pi;
                 kl = loc(k, l);
                 a = siga*wdry[l-1]/dendry[l-1];
                 if(k==l)
                   {    bb = 1.0 - exp(-a);

              /* Changed 5/16/00,  to fix that divide by 0 problem */
                       if ( bb < 1e-30 )
                         bb = 1e-30;
              /* Changed ............................. */

                   area[k-1]=bb;
                   }
                else     /* if(k!=1) */
                 bb = Min(1.0, a);
                   xmat[kl]=bb;
           }
      }
      if(number==1)
      {  elam[0][0]=xmat[1];
                 alone[0]=1.0- elam[0][0];

                 return;
         }
      for(k=1; k<=number; k++)   /*  do k = 1 , number */
      {  frac=0.0;
                 for(l=1; l<=k; l++)     /*  do l = 1 , k */
           {     kl=loc(k, l);
                 frac+=xmat[kl];
           }
                 if(frac>1.0)
           {     for(l=1; l<=k; l++)     /* do l = 1 , k */
                 {       kl=loc(k, l);
                 elam[k-1][l-1]=xmat[kl]/frac;
                }
                   alone[k-1]=0.0;
           }
                 else
           {     for(l=1; l<=k; l++)     /*   do l = 1 , k */
                 {       kl=loc(k, l);
                 elam[k-1][l-1]=xmat[kl];
                   }
              alone[k-1]=1.0-frac;
              }
      }
 }


/*------------------------------------------------------------------------------
//
//      subroutine HEATX()
//  Given horizontal windspeed u at height d [top of fuelbed], cylindrical
//  fuel particle diameter dia, fire environment temperature tf, and mean
//  surface temperature, ts, subroutine returns film heat transfer coefficient
//  hfm and an "effective" film heat transfer coefficient including radiation
//  heat transfer, hbar.  Using the wood's thermal conductivity, cond, the
//  modified Nusselt number [ en ] used to estimate onset of surface drying
//  is returned as well.
//
//-------------------------------------------------------------------------- */
void HeatExchange(double dia , double tf , double ts ,
                  double *hfm , double *hbar , double cond, double *en)
{
double v, re, enuair, conair, fac, hfmin;
const double g = 9.8;
const double vis = 7.5e-05;
const double a = 8.75e-03;
const double b = 5.75e-05;
const double rad = 5.67e-08;
const double fmfac = 0.382;
const double hradf = 0.5;
double hrad;

   *hfm = 0.0;
   if ( dia > b ) {
     v = sqrt(u*u+0.53*g*d);
     re = v*dia/vis;
     enuair = 0.344*pow(re, 0.56);
     conair = a+b*tf;
     fac = sqrt(fabs(tf-ts)/dia);
     hfmin = fmfac*sqrt(fac);
     *hfm = Max((enuair*conair/dia), hfmin);  }
   hrad = hradf*rad*(tf+ts)*(tf*tf+ts*ts);
  *hbar = *hfm + hrad;
  *en = *hbar * dia / cond;
 }


/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
//  wodot , ash , htval , maxno , number , maxkl , area , fint , fi )
//  Computes fi = site avg fire intensity given the burning rates of all
//  interacting pairs of fuel components [ wodot ] , the mineral ash content
//  of each component [ ash ] , the heat of combustion value [ htval ] for
//  each , and the number of fuel components [ number ] , where max = maxno.
//  fi is in kW / sq m , while htval is in J / kg.
//  fint( k ) is the correction to fi to adjust
//  the intensity level to be the local value where size k is burning.
//
//  Out: ad_pcSmoCon......percent that was consumed in Smoldering
//  Ret: Fire Intensity
.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
double FireIntensity(double *ad_pcSmoCon)
{

long k, l, kl, k0;
double sum, wdotk, ark, term;
double wnoduff, wnoduffsum, noduffsum, test, noduffterm;  /*, fintnoduff[MAXNO]; */
double fracf;
double d, d_TotFla, d_TotSmo;
char  cr[100]; 

    FIC_Init ();

// test
   _CompDump (0, 0, 0, "Init", 0, 0);  /* Init for Test Dump file - not actually needed by burnup to run */ 
// test

     d_TotFla = d_TotSmo = 0;

     if ( gd_Fudge1 != 0 ) {                 /* A kludge to save what the    */
          wodot[0]= gd_Fudge1;               /* 'Start' function was trashing*/
          gd_Fudge1 = 0; }                   /* This will only happen on the */
     if ( gd_Fudge2 != 0 ) {                 /* first time step              */
          wodot[1]= gd_Fudge2;
          gd_Fudge2 = 0; }

     sum = noduffsum = wnoduffsum = 0.0;
     for ( k = 1; k <= number; k++) {
       wdotk = wnoduff = 0.0;
       for ( l = 0; l <= k; l++) {
          kl = loc(k, l);
          wdotk += wodot[kl]; }

       term = (1.0-ash[k-1])*htval[k-1]*wdotk*1.e-03;
       ark = area[k-1];
       if ( ark > e_small)
          fint[k-1] = term / ark - term;
       else
          fint[k-1] = 0.0;

// F-62 completion/acceptance-recovery pass (2026-09-21) DIAGNOSTIC HOOK.
// Same opt-in env var as _CompDump()/Start()'s own hooks above -- dumps
// the raw wodot[]/ash/htval/term inputs FireIntensity() itself just used
// for k==1 (Litter, the smallest-diameter/highest-sigma sorted class),
// to a separate file, on EVERY call (not just when the class classifies
// "F"/"S", unlike _CompDump). Purely observational -- reads only.
if ( k == 1 && getenv("FOFEM_TEST_BURN_DIAG") != NULL ) {
  FILE *fhFI = fopen ("_FIDump.csv", "a");
  if ( fhFI != NULL ) {
    static bool wroteFIHeader = false;
    if ( !wroteFIHeader ) {
      fprintf (fhFI, "call_idx,k,wdotk,ash,htval,term,ark,fint\n");
      wroteFIHeader = true;
    }
    static long fi_call_idx = 0;
    fprintf (fhFI, "%ld,%ld,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
             fi_call_idx, k, wdotk, ash[k-1], htval[k-1], term, ark, fint[k-1]);
    fi_call_idx++;
    fclose (fhFI);
  }
}

       k0 = loc(k, 0);
       wnoduff = wdotk - wodot[k0];

       noduffterm = (1.0 - ash[k-1]) * htval[k-1] * wnoduff * 1.e-03;
       if ( wnoduff > 0.0) {
         fracf = wnoduff / wdotk;
         test = fracf * fint[k-1];  }
       else
         test = 0.0;

 /*-------------------------------------- */
 /*      flaming and smoldering decision here */
 /*-------------------------------------- */

//   Original -->    if ( test > 15.0 / ark - 15.0) {        /* 15 kW m-2 is switch */
// changed to >= to make litter load over 11.4 (approx) to avoid it smoldering
       if ( test >= 15.0 / ark - 15.0) {        /* 15 kW m-2 is switch */
          d_TotFla += wnoduff;
          strcpy (cr,"F"); }
       else {
          d_TotSmo += wnoduff; 
          strcpy (cr, "S"); }
 /*-------------------------------------- */

/* Save indivual componets tells if flame/smolder, don't really need */
/*  'term, 'test' 'wonoduff (isn't the consumed load) */
  FIC_Put(sigma[k-1],dendry[k-1], term, test, wnoduff, cr); 

/* dump file for testing - see _CompDump() to turn it on */
double x = 15.0 / ark - 15.0;
 _CompDump (test, x, term, cr, wnoduff, sigma[k-1]); 

       sum += term;
       noduffsum += noduffterm;
       wnoduffsum += wnoduff;
    }

  d = d_TotFla + d_TotSmo;                   /* Total Consumed               */
  if ( d != 0 )                              /* don't div by 0               */
     *ad_pcSmoCon = d_TotSmo / d;            /* what % of tot con was Smoldr */
  else
     *ad_pcSmoCon = 0;


// test....................
 _CompDump (sum, 0, 0, "Put", d, *ad_pcSmoCon);
// test....................

   return  sum;
}





/*-----------------------------------------------------------------------------
//
// subroutine DRYTIM( enu , theta , tau )
//  Given a Nusselt number ( enu , actually Biot number = h D / k )
//  and the dimensionless temperature rise required for the start
//  of surface drying ( theta ), returns the dimensionless time ( tau )
//  needed to achieve it.  The time is given multiplied by thermal
//  diffusivity and divided by radius squared.  Solution by binary search.
//
//-------------------------------------------------------------------------- */
double DryTime(double enu, double theta)
{
long n;
double tau;
double x, xl, xh, xm;
const double p = 0.47047;

    xl = 0.0;
    xh = 1.0;
    for ( n = 0; n < 15; n++) {
      xm = 0.5 * ( xl + xh );
      if ( func(xm, theta) < 0.0)
        xl = xm;
      else
        xh = xm; }
    x = (1.0 / xm - 1.0 ) / p;
    tau = pow2(0.5 * x / enu);
    return tau;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Heat_Heading
* Desc: Heading for the .hea file
*   In: fh......file handle
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
void Heat_Heading (FILE  *fh)
{
   fprintf (fh, " Column  1: Time since ignition, seconds\n");
   fprintf (fh, "         2: Intensity  (%s)\n", e_KwSqM);
   fprintf (fh, "         3: Emission  PM 2.5   (%s)\n", e_GrSqM);
   fprintf (fh, "         4: Emission   PM 10   (%s)\n", e_GrSqM);
   fprintf (fh, "         5: Emission     CH4   (%s)\n", e_GrSqM);
   fprintf (fh, "         6: Emission     CO2   (%s)\n", e_GrSqM);
   fprintf (fh, "         7: Emission      CO   (%s)\n", e_GrSqM);

   fprintf (fh, "         8: Emission     NOX   (%s)\n", e_GrSqM);
   fprintf (fh, "         9: Emission     SO2   (%s)\n", e_GrSqM);


// changed 3-24-16 Ticket #62
//   fprintf (fh, "        10: Flaming & Short Term Smoldering Weight    (%s)\n", e_KgSqM);
//   fprintf (fh, "        11: Residual Smoldering Weight                (%s)\n\n", e_KgSqM);

     fprintf (fh, "        10: Flaming Weight     (%s)\n", e_KgSqM);
     fprintf (fh, "        11: Smoldering Weight  (%s)\n", e_KgSqM);
     fprintf (fh, "        12: Flame Weight       (%s)\n", e_TonAc);
     fprintf (fh, "        13: Smoldering Weight  (%s)\n\n", e_TonAc);


}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Heat_HeadingFS
* Desc: Heading for the .hea file - Flaming and Smoldering
*   In: fh......file handle
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
void Heat_HeadingFS (FILE  *fh)
{
   fprintf (fh, " Column     1: Time since ignition, (%s)\n",e_Second);
   fprintf (fh, "            2: Intensity  (%s)\n", e_KwSqM);
   fprintf (fh, "   \n");
   fprintf (fh, "          3-4: Emission PM 2.5   flame/smolder  (%s)\n", e_GrSqM);
   fprintf (fh, "          5-6: Emission   PM 10  flame/smolder  (%s)\n", e_GrSqM);
   fprintf (fh, "          7-8: Emission     CH4  flame/smolder  (%s)\n", e_GrSqM);
   fprintf (fh, "         9-10: Emission     CO2  flame/smolder  (%s)\n", e_GrSqM);
   fprintf (fh, "        11-12: Emission      CO  flame/smolder  (%s)\n", e_GrSqM);

   fprintf (fh, "        13-14: Emission     NOX  flame/smolder  (%s)\n", e_GrSqM);
   fprintf (fh, "        15-16: Emission     SO2  flame/smolder  (%s)\n", e_GrSqM);
   fprintf (fh, "           17: Flame Weight       (%s)\n", e_KgSqM);
   fprintf (fh, "           18: Smoldering Weight  (%s)\n\n", e_KgSqM);
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: DuffBurn
* Desc: Fran
* NOTE:
*  9-18-2015 - ran test...
* DL pulled this equation from Frandsen literature, it looks different than
*  the equation we use below, BUT it produces the same result
* tdf = ff * wdf / (7.5 x 10-4 � 2.7 x 10-4*Rm)
*                 
* Note-1: This function has been modified to use the Duff Consumed Percent
*          that fofem calculates.
*         But if a negative or and out of range percent comes in the function
*         does the calculation the orginal way, this way should get used
*         when running from a user Burnup input file, because burnup will
*         run strictly from the input file, with fofem having not been run
*         there is no duff con percent available.
* ---> SEE COMMENT for input Param below
*  Orginal Comment -> Duff burning rate (ergo, intensity) and duration
*   In: f_DufConPerCent.....Duff Consumed Percent Fofem calculation
*                           as a whole number, we convert it in here
*                           NOTE, send in a -1 when running burnup with the
*                           'optional' user input file
*       wdf......Duff Dry Weight kilograms per cubic meter
*       dfm......Duff Moisture - decial percent, 0 -> 1.96 
*
*  Out: *dfi..........Duff Fire Intensity kW Kilowatts
*       *tdf..........Amount of Time (Seconds) Duff will burn out
*       *ad_Duf_CPTS..Amount of Duff Consumed Per Time Step
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
void DuffBurn (double wdf, double dfm, double *dfi, double *tdf,
                       float f_DufConPerCent, double *ad_Duf_CPTS)
{
double ff;
   *dfi = 0.0;
   *tdf = 0.0;
   *ad_Duf_CPTS = 0.0;

/* Note: all check of limits should have been done before coming here        */

   if (( wdf <= 0.0) || ( dfm >= 1.96))
      return;

   *dfi=11.25-4.05*dfm;

#ifdef FOFEM
/* Compiling for the FOFEM GUI we use Duff Type selected on the Settings Window */ 
    *dfi = _FireInt_DuffType (dfm);
#endif 

/* changed 6-3-04, ff = f_DufConPerCent to divide by 100.0, it was coming    */
/*   in as a whole number and using that original equation in the 'else'     */
   if ( f_DufConPerCent >= 0 &&  f_DufConPerCent <= 100 )
     ff = f_DufConPerCent / 100.0;  /* See Note-1 in function header         */
   else
     ff=0.837-0.426*dfm;            /* NOTE: this line isOrginal Code        */

/*  Time (seconds) to Burn Duff is bassed on - percent consumed & duff weight & duff moisture */
   *tdf=1.e+04*ff*wdf/(7.5-2.7*dfm); 

/* line added 11-7-00, when we had the problem not get duff in calcs */
/* I put in the check for div by 0, which can happen */
   if ( *tdf == 0 ) {
     *ad_Duf_CPTS = 0; }                     /* Duf Consmed per second    */
   else {
     *ad_Duf_CPTS  = (ff * wdf) / ( *tdf );} /* Duf Consmed per second    */
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Get_Cons
* Desc: Take the difference of the old new consumed values.
*       This gets called from every place in 'Start' and 'Step' functions
*       where the consumed amount is calculated.
* NOTE: The way Burnup works is that it replaces the old remaining load
*        with  the new load, so we take the difference to get the consumed
*   In: d_old, d_new.....old and new remaingin loads
*  Ret: the consumed amount whick is the difference
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
double Get_Cons (double d_old, double d_new)
{
  return (d_old - d_new);
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Duff_CPTS
* Desc: Calc the Duff Consumed per time step.
*       This also adjusts the total Duff that is remaining to be consumed
*       This function maintains the Total amount of Duff that has not be
*        used yet, so when the Total gets down to or near 0 the function
*        will return 0 or the small remaining amount.
*   In: d_Duf_CPS.....duff to be consumed per second
*       d_timeSec.....number of seconds
* In/Out: *ad_Duf_ConTot.....running amount of Total Duff, whatever is
*                            calculated and returned is deducted from here
*  Ret:  Amount Consumed.
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
double Duff_CPTS ( double *ad_Duf_ConTot, double d_Duf_CPT, double d_timeSec)
{

double d_AmtCon;
  if ( *ad_Duf_ConTot == 0 )                 /* Nothing to left in total    */
     return 0;
  if ( d_Duf_CPT == 0 )                      /* watch out                    */
     return 0;

  d_AmtCon = d_Duf_CPT * d_timeSec;          /* Calc amount                  */
  if ( *ad_Duf_ConTot >= d_AmtCon ) {        /* make sure have enoug in tot  */
    *ad_Duf_ConTot = *ad_Duf_ConTot - d_AmtCon; /* reduce tot                */
     return d_AmtCon; }                      /* ret amout                    */

  d_AmtCon = *ad_Duf_ConTot;                 /* not enough so ret whats left */
  *ad_Duf_ConTot = 0;                        /* 0 the tot                    */
  return d_AmtCon;

}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ES_Calc
* -----> Revised 8-25-18
*        Stopped using d_pcSmo (from FireIntensity()) and started using the
*        individual fuel component flame/smolder info acquired in 
*        FireIntensity
* Desc: Calculate the Emissions and save totals (keeps running totals)
*       This function gets called for each time step.
*       The amounts and totals are put in the ES (Emission Struct)
* NOTE: The ES struct has to be initialized 'ES_Init' function before
*        using so final total variables in the struct get zero'd
* 
*
* Note-2: 8-20-18 start using the percent smoldering calculated from the
*          individual consumed amounts - insted of values passed into 
*          this function
* Note-3: Looking at the indivual component loads and the Wood Liter amount
*         coming into this function (both from FireIntensity) seem to be 
*         the same, but using the indivual amounts is a better reflection
*         or whats in flaming or smolder, verse the overall percent
*         smoldering that gets passed in here
* NOTE: input consumed amounts are ---> Kilograms per square meter
*       emissions are calculated ----> Grams per square meter
*
*   In: d_WooLit.....Wood and Litter consumed for current time step
*
*       d_Duff.......Duff consumed for current time step
*       d_HSFB.......Herb,Shrub,Foliage,Branch, comsumed amounts
* 
* NOTE: Not Used with new revised procedure 
*       d_pcSmo......percent of Wood/Litter load that was consumed by
*                     smoldering
*
*       d_time.......time in seconds of time step, 60,75,90, etc..
* In/Out: a_ES........The emission amounts for the time step are placed
*                      in here, also they are added to the running totals
*                      that are alss in this struct
*  Ret: 1 OK - 0 Error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int  ES_Calc (d_ES *a_ES, double d_WooLit,  double d_Duff, double d_HSFB,
               double d_pcSmo, double d_time)
{
double  d,dd, d_pcFla;
double  d_FlaCon, d_SmoCon;

float f,g, s, f_SmoWL, f_FlaWL, f_Smo1k, f_Fla1k;


   if ( d_pcSmo < 0 || d_pcSmo > 1.0 ) {
     return 0; }

double d_Coarse;
float  f_ConTPA;

/* Set the individual fuel loads and smolder percent */
   d_Coarse = ES_SetComponents (0, &f_ConTPA,d_time, a_ES); /* Amt 1k coarse wood consumed */
 
/* Get wood lit consumed in smolder and Flame */
   ES_FlaSmo (a_ES, &f_SmoWL, &f_FlaWL, &f_Smo1k, &f_Fla1k);


/* 8-20-18 - See Note-2 above */ 
   d_pcSmo = a_ES->d_cn_SmoConPC; 
   d_pcFla = 1.0 - d_pcSmo;                  /* % Consmd Flam, see Note-1  */
   a_ES->d_pcSmo = d_pcSmo; 

/* See Note-3 Above */ 
//   d_FlaCon = ( d_WooLit * d_pcFla ) + d_HSFB;   /* Tot Consumed by Flam     */
//   d_SmoCon = ( d_WooLit * d_pcSmo ) + d_Duff;   /* Tot Consumed by Smoldrng */ 
   d_FlaCon = f_FlaWL + f_Fla1k + d_HSFB;  /* Flame wood lit 1k + herb shr fol bra*/ 
   d_SmoCon = f_SmoWL + f_Smo1k + d_Duff;  /* Smolder wood lit 1k */
  
   a_ES->d_FlaCon  = d_FlaCon;    /* Wood Lit HSFB  */
   a_ES->d_SmoCon  = d_SmoCon;    /* Wood Lit Duff */ 
   a_ES->d_DuffCon = d_Duff; 
   a_ES->d_HSFBCon = d_HSFB; 


/* Consumed Running totals for Flame and Smoldering */
   a_ES->dN_FlaCon  += a_ES->d_FlaCon;     /* for getting Total Consumed      */
   a_ES->dN_SmoCon  += a_ES->d_SmoCon;
   a_ES->dN_DuffCon += a_ES->d_DuffCon;  /* Accumlated total */
   a_ES->dN_HSFBCon += a_ES->d_HSFBCon; /* Herb Shr, fol bra */

/* Emissions each time step */
   a_ES->d_PM25F = d_pm25f * d_FlaCon;
   a_ES->d_PM10F  =d_pm10f * d_FlaCon;
   a_ES->d_CH4F  = d_ch4f  * d_FlaCon;
   a_ES->d_CO2F  = d_co2f  * d_FlaCon;
   a_ES->d_COF   = d_cof   * d_FlaCon;
   a_ES->d_NOXF  = d_noxf  * d_FlaCon;
   a_ES->d_SO2F  = d_so2f  * d_FlaCon;
                                   
   a_ES->d_PM25S = d_pm25s * d_SmoCon;
   a_ES->d_PM10S = d_pm10s * d_SmoCon;
   a_ES->d_CH4S  = d_ch4s  * d_SmoCon;
   a_ES->d_CO2S  = d_co2s  * d_SmoCon;
   a_ES->d_COS   = d_cos   * d_SmoCon;
   a_ES->d_NOXS  = d_noxs  * d_SmoCon;
   a_ES->d_SO2S  = d_so2s  * d_SmoCon;  

/* Just Duff - NOTE the above smoldering emission INCLUDE duff */
   a_ES->d_PM25S_Duff = d_pm25s * d_Duff; 
   a_ES->d_PM10S_Duff = d_pm10s * d_Duff; 
   a_ES->d_CH4S_Duff  = d_ch4s  * d_Duff; 
   a_ES->d_CO2S_Duff  = d_co2s  * d_Duff; 
   a_ES->d_COS_Duff   = d_cos   * d_Duff; 
   a_ES->d_NOXS_Duff  = d_noxs  * d_Duff; 
   a_ES->d_SO2S_Duff  = d_so2s  * d_Duff; 

               
/* Emissions each time step - combined flame + smolder */
   a_ES->d_PM25 = a_ES->d_PM25F + a_ES->d_PM25S;
   a_ES->d_PM10 = a_ES->d_PM10F + a_ES->d_PM10S;
   a_ES->d_CH4  = a_ES->d_CH4F  + a_ES->d_CH4S; 
   a_ES->d_CO2  = a_ES->d_CO2F  + a_ES->d_CO2S; 
   a_ES->d_CO   = a_ES->d_COF   + a_ES->d_COS;  
   a_ES->d_NOX  = a_ES->d_NOXF  + a_ES->d_NOXS; 
   a_ES->d_SO2  = a_ES->d_SO2F  + a_ES->d_SO2S; 

/* Emission Running Totals for Flame & Smoldering.....................................  */
   a_ES->dN_PM25F +=  a_ES->d_PM25F;
   a_ES->dN_PM10F +=  a_ES->d_PM10F;
   a_ES->dN_CH4F  +=  a_ES->d_CH4F; 
   a_ES->dN_CO2F  +=  a_ES->d_CO2F; 
   a_ES->dN_COF   +=  a_ES->d_COF;  
   a_ES->dN_NOXF  +=  a_ES->d_NOXF; 
   a_ES->dN_SO2F  +=  a_ES->d_SO2F; 
                                   
   a_ES->dN_PM25S +=  a_ES->d_PM25S;
   a_ES->dN_PM10S +=  a_ES->d_PM10S;
   a_ES->dN_CH4S  +=  a_ES->d_CH4S; 
   a_ES->dN_CO2S  +=  a_ES->d_CO2S; 
   a_ES->dN_COS   +=  a_ES->d_COS;  
   a_ES->dN_NOXS  +=  a_ES->d_NOXS; 
   a_ES->dN_SO2S  +=  a_ES->d_SO2S; 

/* Emission Run Total for JUST DUFF - NOTE above Smolder includes Duff */
   a_ES->dN_PM25S_Duff   +=   a_ES->d_PM25S_Duff; 
   a_ES->dN_PM10S_Duff   +=   a_ES->d_PM10S_Duff; 
   a_ES->dN_CH4S_Duff    +=   a_ES->d_CH4S_Duff;  
   a_ES->dN_CO2S_Duff    +=   a_ES->d_CO2S_Duff;  
   a_ES->dN_COS_Duff     +=   a_ES->d_COS_Duff;   
   a_ES->dN_NOXS_Duff    +=   a_ES->d_NOXS_Duff;  
   a_ES->dN_SO2S_Duff    +=   a_ES->d_SO2S_Duff;  

   a_ES->dN_Cnt++; 

   if ( a_ES->d_FlaCon > 0.00001 )           /* Save last time step that we  */
     a_ES->d_FlaDur = d_time;                /* have a consumed amount       */
   if ( a_ES->d_SmoCon > 0.00001 ) {         /* See Note-1 above             */
     a_ES->d_SmoDur = d_time; }
   return 1;
}


#ifdef OLDOLD
/**************************************************************************
*
*
* Note-1: Herb,shrb,foliage,branch will only come in on first 60 second
*          time step NOTE - We Changed that june 2015
* Note-2: Check to see if the amount of coarse wood consumed is greater 
*         than the total coarse + fine + litter consumed, logically this
*         shouldn't happen because the amounts are calculated from the same
*         arrays. But the difference is so small, see the check i put in,
*         So not sure what is causing the diff but I just set the amount
*         of coarse wood consumed to the total wood-lit consumed so it
*         doesn't screw things up.
* Note-3: Once the fire intensity drops below the critical point ( 15 kw )
*          we consider it smoldering, DL said to put this check in becasue
*          the flaming time was really long when we just depended on there
*          being fine fuels still burning and using their Flame emis factor
*         
**************************************************************************/
int  ES_Calc_NEW (d_ES *a_ES, double d_WooLit,  double d_Duff, double d_HSFB, double d_time, float f_FirInt)
{
float f_ConTPA;
double  d_pcFla;
double  d_pm10, d_pm25, d_ch4, d_co, d_co2, d_nox,  d_so2;
double  d_TotCon;

double d, d_Flame, d_Coarse, d_Fine, d_Smolder;


/* Clear all the Single Step totals */
   a_ES->d_PM25 = a_ES->d_PM25F = a_ES->d_PM25S = 0;
   a_ES->d_CH4  = a_ES->d_CH4F  = a_ES->d_CH4S  = 0;
   a_ES->d_CO   = a_ES->d_COF   = a_ES->d_COS   = 0;
   a_ES->d_CO2  = a_ES->d_CO2F  = a_ES->d_CO2S  = 0;
   a_ES->d_PM10 = a_ES->d_PM10F = a_ES->d_PM10S = 0;
   a_ES->d_NOX  = a_ES->d_NOXF  = a_ES->d_NOXS  = 0;
   a_ES->d_SO2  = a_ES->d_SO2F  = a_ES->d_SO2S  = 0;


/* Deal with Duff first...................... */
   a_ES->d_PM25  += gf_dPM25 * d_Duff;
   a_ES->d_CH4   += gf_dCH4  * d_Duff;
   a_ES->d_CO    += gf_dCO   * d_Duff;
   a_ES->d_CO2   += gf_dCO2  * d_Duff;
   a_ES->d_PM10  += gf_dPM10 * d_Duff;
   a_ES->d_NOX   += gf_dNOX  * d_Duff;
   a_ES->d_SO2   += gf_dSO2  * d_Duff;

   a_ES->d_PM25S += gf_dPM25 * d_Duff;
   a_ES->d_CH4S  += gf_dCH4  * d_Duff;
   a_ES->d_COS   += gf_dCO   * d_Duff;
   a_ES->d_CO2S  += gf_dCO2  * d_Duff;
   a_ES->d_PM10S += gf_dPM10 * d_Duff;
   a_ES->d_NOXS  += gf_dNOX  * d_Duff;
   a_ES->d_SO2S  += gf_dSO2  * d_Duff;

   a_ES->dN_PM25S += gf_dPM25 * d_Duff;    /* add to final totals */ 
   a_ES->dN_CH4S  += gf_dCH4  * d_Duff;
   a_ES->dN_COS   += gf_dCO   * d_Duff;
   a_ES->dN_CO2S  += gf_dCO2  * d_Duff;
   a_ES->dN_PM10S += gf_dPM10 * d_Duff;
   a_ES->dN_NOXS  += gf_dNOX  * d_Duff;
   a_ES->dN_SO2S  += gf_dSO2  * d_Duff;


/*----------------------------------------------------------*/
/* Flaming Fuels */
   d_Coarse = ES_SetComponents (0, &f_ConTPA,d_time, a_ES); /* Amt 1k coarse wood consumed */
   d_Flame = d_HSFB;                         /* Herb,shr,fol,bra see note-1 above */

/* 1000 Hr Flame and Smolder - used in output .csv  consumed - emission files */
   if ( f_FirInt > gf_CriInt ) {
      a_ES->d_cn_1kHrFla = d_Coarse;
      a_ES->d_cn_1kHrSmo = 0 ; }
//      gf_1kFlame += d_Coarse;
   else {
      a_ES->d_cn_1kHrSmo = d_Coarse;
      a_ES->d_cn_1kHrFla = 0; }
//    gf_1kSmold += d_Coarse;


   if ( d_Coarse > d_WooLit ) {     /* See Note-2 above */ 
     d_Coarse = d_WooLit; 
     d = d_Coarse - d_WooLit;       /* a test to see how much diff there is */
     if ( d > 0.001 )               /* and it's not very much at all */
	    d = 0; 
   }

   if ( f_FirInt > gf_CriInt ) {   /* burnup fire intensity - flame smolder point */
     d_Flame += d_WooLit;          /*  coarse, fine, litter - all going up in flame */
     d_Coarse = 0; }
   else {
     d = d_WooLit - d_Coarse;      /* separate coarse from fine/litter */
     d_Flame += d; }               /*  coarse will go to smolder below */

   a_ES->d_PM25  +=  gf_fPM25 * d_Flame;
   a_ES->d_CH4   +=  gf_fCH4  * d_Flame;
   a_ES->d_CO    +=  gf_fCO   * d_Flame;
   a_ES->d_CO2   +=  gf_fCO2  * d_Flame;
   a_ES->d_PM10  +=  gf_fPM10 * d_Flame;
   a_ES->d_NOX   +=  gf_fNOX  * d_Flame;
   a_ES->d_SO2   +=  gf_fSO2  * d_Flame;

   a_ES->d_PM25F =  gf_fPM25 * d_Flame;
   a_ES->d_CH4F  =  gf_fCH4  * d_Flame;
   a_ES->d_COF   =  gf_fCO   * d_Flame;
   a_ES->d_CO2F  =  gf_fCO2  * d_Flame;
   a_ES->d_PM10F =  gf_fPM10 * d_Flame;
   a_ES->d_NOXF  =  gf_fNOX  * d_Flame;
   a_ES->d_SO2F  =  gf_fSO2  * d_Flame;

   a_ES->dN_PM25F += gf_fPM25 * d_Flame;
   a_ES->dN_CH4F  += gf_fCH4  * d_Flame;
   a_ES->dN_COF   += gf_fCO   * d_Flame;
   a_ES->dN_CO2F  += gf_fCO2  * d_Flame;
   a_ES->dN_PM10F += gf_fPM10 * d_Flame;
   a_ES->dN_NOXF  += gf_fNOX  * d_Flame;
   a_ES->dN_SO2F  += gf_fSO2  * d_Flame;

   a_ES->dN_Cnt++; 

/*-------------------------------------------*/
/* Coarse Woody is smoldering                */
   a_ES->d_PM25  +=  gf_sPM25 * d_Coarse;
   a_ES->d_CH4   +=  gf_sCH4  * d_Coarse;
   a_ES->d_CO    +=  gf_sCO   * d_Coarse;
   a_ES->d_CO2   +=  gf_sCO2  * d_Coarse;
   a_ES->d_PM10  +=  gf_sPM10 * d_Coarse;
   a_ES->d_NOX   +=  gf_sNOX  * d_Coarse;
   a_ES->d_SO2   +=  gf_sSO2  * d_Coarse;

   a_ES->d_PM25S +=  gf_sPM25 * d_Coarse;
   a_ES->d_CH4S  +=  gf_sCH4  * d_Coarse;
   a_ES->d_COS   +=  gf_sCO   * d_Coarse;
   a_ES->d_CO2S  +=  gf_sCO2  * d_Coarse;
   a_ES->d_PM10S +=  gf_sPM10 * d_Coarse;
   a_ES->d_NOXS  +=  gf_sNOX  * d_Coarse;
   a_ES->d_SO2S  +=  gf_sSO2  * d_Coarse;

   a_ES->dN_PM25S  += gf_sPM25 * d_Coarse;
   a_ES->dN_CH4S   += gf_sCH4  * d_Coarse;
   a_ES->dN_COS    += gf_sCO   * d_Coarse;
   a_ES->dN_CO2S   += gf_sCO2  * d_Coarse;
   a_ES->dN_PM10S  += gf_sPM10 * d_Coarse;
   a_ES->dN_NOXS   += gf_sNOX  * d_Coarse;
   a_ES->dN_SO2S   += gf_sSO2  * d_Coarse;
   

   a_ES->d_FlaCon = d_Flame;                /* Store amounts used in calcs  */
   a_ES->dN_FlaCon += a_ES->d_FlaCon;       /* for getting Total Consumed      */  
 
 
   if ( f_FirInt > gf_CriInt ) {            /* See Note-3 above */
     if ( a_ES->d_FlaCon > 0.00001 )        /* Save last time step that we  */
      a_ES->d_FlaDur = d_time;              /* have a consumed amount       */ 
   }

   d_Smolder = d_Duff + d_Coarse;
   a_ES->d_SmoCon = d_Smolder;
   a_ES->dN_SmoCon += a_ES->d_SmoCon;
  
   if ( a_ES->d_SmoCon > 0.00001 )       
     a_ES->d_SmoDur = d_time; 

/* Save Consumed Amounts */
   a_ES->d_Flame = d_Flame;
   a_ES->d_Coarse = d_Coarse;  
   a_ES->d_Duff = d_Duff;

/* Grand totals, adding these up incase I need them, */ 
   a_ES->dN_Flame  += d_Flame;
   a_ES->dN_Coarse += d_Coarse;  
   a_ES->dN_Duff   += d_Duff;
   
   return 1 ; 
}
#endif 

// 888888888888888888888888888888888888888888888888888888888
/**************************************************************************
* Name: ES_Calc_NEW
* Desc: Expanded Emissions
* NOTE: Revised Aug 25 2018
*       Flame and Smolder point is now based on the indiviual fuel compoents 
*       flame/smolder point decission in the FireIntensity() function.
*       Previously, all small wood/lit was flame/residual smolder, 1k had
*       to be over 15 kw/m2 for flaming
**************************************************************************/
int  ES_Calc_NEW (d_ES *a_ES, double d_WooLit,  double d_Duff, double d_HSFB, double d_time, float f_FirInt)
{
float f_ConTPA;
double  d_pcFla;
double  d_pm10, d_pm25, d_ch4, d_co, d_co2, d_nox,  d_so2;
double  d_TotCon;
double d, d_Flame, d_Coarse, d_Fine, d_Smolder;
float  f_SmoCon, f_FlaCon;
float  f_SmoWL, f_FlaWL, f_Smo1k, f_Fla1k;

/* Clear all the Single Step totals */
   a_ES->d_PM25 = a_ES->d_PM25F = a_ES->d_PM25S = 0;
   a_ES->d_CH4  = a_ES->d_CH4F  = a_ES->d_CH4S  = 0;
   a_ES->d_CO   = a_ES->d_COF   = a_ES->d_COS   = 0;
   a_ES->d_CO2  = a_ES->d_CO2F  = a_ES->d_CO2S  = 0;
   a_ES->d_PM10 = a_ES->d_PM10F = a_ES->d_PM10S = 0;
   a_ES->d_NOX  = a_ES->d_NOXF  = a_ES->d_NOXS  = 0;
   a_ES->d_SO2  = a_ES->d_SO2F  = a_ES->d_SO2S  = 0;

/*----------------------------------------------------------*/
/* set individual wood & lit fuel loads and percent smolder */
   d_Coarse = ES_SetComponents (0, &f_ConTPA,d_time, a_ES);

/* Get wood lit consumed in smolder and Flame */
   ES_FlaSmo (a_ES, &f_SmoWL, &f_FlaWL, &f_Smo1k, &f_Fla1k);


/* use the Flame Emission factors on consumed by fire */
   d_Flame = (f_FlaWL + f_Fla1k + d_HSFB);  /* small Wood, Lit, 1000k flame, herb shr fol bra*/ 

   a_ES->d_PM25  +=  gf_fPM25 * d_Flame;   /* */
   a_ES->d_CH4   +=  gf_fCH4  * d_Flame;
   a_ES->d_CO    +=  gf_fCO   * d_Flame;
   a_ES->d_CO2   +=  gf_fCO2  * d_Flame;
   a_ES->d_PM10  +=  gf_fPM10 * d_Flame;
   a_ES->d_NOX   +=  gf_fNOX  * d_Flame;
   a_ES->d_SO2   +=  gf_fSO2  * d_Flame;

   a_ES->d_PM25F =  gf_fPM25 * d_Flame;
   a_ES->d_CH4F  =  gf_fCH4  * d_Flame;
   a_ES->d_COF   =  gf_fCO   * d_Flame;
   a_ES->d_CO2F  =  gf_fCO2  * d_Flame;
   a_ES->d_PM10F =  gf_fPM10 * d_Flame;
   a_ES->d_NOXF  =  gf_fNOX  * d_Flame;
   a_ES->d_SO2F  =  gf_fSO2  * d_Flame;

   a_ES->dN_PM25F += gf_fPM25 * d_Flame;
   a_ES->dN_CH4F  += gf_fCH4  * d_Flame;
   a_ES->dN_COF   += gf_fCO   * d_Flame;
   a_ES->dN_CO2F  += gf_fCO2  * d_Flame;
   a_ES->dN_PM10F += gf_fPM10 * d_Flame;
   a_ES->dN_NOXF  += gf_fNOX  * d_Flame;
   a_ES->dN_SO2F  += gf_fSO2  * d_Flame;

/* Do  Sm1000 Hr - small wood liter smoldering emiss factors - consumed */
   d_Smolder = f_Smo1k + f_SmoWL; 
   a_ES->d_PM25  +=  gf_sPM25  * d_Smolder;
   a_ES->d_CH4   +=  gf_sCH4   * d_Smolder;
   a_ES->d_CO    +=  gf_sCO    * d_Smolder;
   a_ES->d_CO2   +=  gf_sCO2   * d_Smolder;
   a_ES->d_PM10  +=  gf_sPM10  * d_Smolder;
   a_ES->d_NOX   +=  gf_sNOX   * d_Smolder;
   a_ES->d_SO2   +=  gf_sSO2   * d_Smolder;

   a_ES->d_PM25S +=  gf_sPM25  * d_Smolder;
   a_ES->d_CH4S  +=  gf_sCH4   * d_Smolder;
   a_ES->d_COS   +=  gf_sCO    * d_Smolder;
   a_ES->d_CO2S  +=  gf_sCO2   * d_Smolder;
   a_ES->d_PM10S +=  gf_sPM10  * d_Smolder;
   a_ES->d_NOXS  +=  gf_sNOX   * d_Smolder;
   a_ES->d_SO2S  +=  gf_sSO2   * d_Smolder;

   a_ES->dN_PM25S  += gf_sPM25 * d_Smolder;
   a_ES->dN_CH4S   += gf_sCH4  * d_Smolder;
   a_ES->dN_COS    += gf_sCO   * d_Smolder;
   a_ES->dN_CO2S   += gf_sCO2  * d_Smolder;
   a_ES->dN_PM10S  += gf_sPM10 * d_Smolder;
   a_ES->dN_NOXS   += gf_sNOX  * d_Smolder;
   a_ES->dN_SO2S   += gf_sSO2  * d_Smolder;

/* Duff - uses its own duff emssion factors, Duff always go to smolder  */
   a_ES->d_PM25  += gf_dPM25 * d_Duff;    /* add to combined total emis (fla smo duff */
   a_ES->d_CH4   += gf_dCH4  * d_Duff;
   a_ES->d_CO    += gf_dCO   * d_Duff;
   a_ES->d_CO2   += gf_dCO2  * d_Duff;
   a_ES->d_PM10  += gf_dPM10 * d_Duff;
   a_ES->d_NOX   += gf_dNOX  * d_Duff;
   a_ES->d_SO2   += gf_dSO2  * d_Duff;

   a_ES->d_PM25S += gf_dPM25 * d_Duff;   /* add it to the smolder amount */
   a_ES->d_CH4S  += gf_dCH4  * d_Duff;
   a_ES->d_COS   += gf_dCO   * d_Duff;
   a_ES->d_CO2S  += gf_dCO2  * d_Duff;
   a_ES->d_PM10S += gf_dPM10 * d_Duff;
   a_ES->d_NOXS  += gf_dNOX  * d_Duff;
   a_ES->d_SO2S  += gf_dSO2  * d_Duff;

   a_ES->dN_PM25S += gf_dPM25 * d_Duff;    /* add to final totals */ 
   a_ES->dN_CH4S  += gf_dCH4  * d_Duff;
   a_ES->dN_COS   += gf_dCO   * d_Duff;
   a_ES->dN_CO2S  += gf_dCO2  * d_Duff;
   a_ES->dN_PM10S += gf_dPM10 * d_Duff;
   a_ES->dN_NOXS  += gf_dNOX  * d_Duff;
   a_ES->dN_SO2S  += gf_dSO2  * d_Duff;


   a_ES->dN_Cnt++; 

/* 1000 Hr Flame and Smolder - used in output .csv  consumed - emission files */
   a_ES->d_cn_1kHrFla = f_Fla1k;
   a_ES->d_cn_1kHrSmo =  f_Smo1k; 

   a_ES->d_FlaCon = d_Flame;    /* All wood, lit and herb shr fol bra */
   a_ES->d_SmoCon = d_Smolder + d_Duff;
   a_ES->d_DuffCon = d_Duff;    /* keep a separted duff amount */ 
   a_ES->d_HSFBCon = d_HSFB; 

   a_ES->dN_FlaCon  += a_ES->d_FlaCon; 
   a_ES->dN_SmoCon  += a_ES->d_SmoCon; 
   a_ES->dN_DuffCon += a_ES->d_DuffCon;
   a_ES->dN_HSFBCon += a_ES->d_HSFBCon; /* Herb Shr, fol bra */


   if ( f_FirInt > gf_CriInt ) {            /* See Note-3 above */
     if ( a_ES->d_FlaCon > 0.00001 )        /* Save last time step that we  */
      a_ES->d_FlaDur = d_time; }             /* have a consumed amount       */ 

   if ( a_ES->d_SmoCon > 0.00001 )       
     a_ES->d_SmoDur = d_time; 

   return 1 ; 
}
/**********************************************************
* Name: ES_FlaSmo
* Desc: Total up consumed amounts of smolder and flame 
*  out: af_Smo...1,10,100, lit
*       af_Fla...1,10,100, lit
*       af_1kSmo...1000 hr smolder
*       af_1kFla...1000 hr flaming 
**********************************************************/
int ES_FlaSmo (d_ES *a, float *af_Smo, float *af_Fla, float *af_1kSmo, float *af_1kFla)
{
float f, s; 

   f = s = 0; 
   s += a->d_cn_Lit * a->d_cn_LitSmoPerCent;
   f += a->d_cn_Lit * (1.0 - a->d_cn_LitSmoPerCent); 

   s += a->d_cn_1Hr * a->d_cn_1HrSmoPerCent;
   f += a->d_cn_1Hr * (1.0 - a->d_cn_1HrSmoPerCent); 

   s += a->d_cn_10Hr * a->d_cn_10HrSmoPerCent;
   f += a->d_cn_10Hr * (1.0 - a->d_cn_10HrSmoPerCent); 

   s += a->d_cn_100Hr * a->d_cn_100HrSmoPerCent;
   f += a->d_cn_100Hr * (1.0 - a->d_cn_100HrSmoPerCent); 



   *af_1kSmo = a->d_cn_1kHr * a->d_cn_1kSmoPerCent;
   *af_1kFla = a->d_cn_1kHr * ( 1.0 - a->d_cn_1kSmoPerCent);

   *af_Smo = s; 
   *af_Fla = f; 

   return 1; 

}
// 888888888888888888888888888888888888888888888888888888888

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ES_*
* Desc: Functions to get final totals and time durations.
*       Once burnup is done the total emissions can be gotten here
*       and Flame & Smoldering Duration, which is the time step in
*       which the emission finally stop (last one).
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/

float ES_PM10F() { return (float) s_ES.dN_PM10F; }
float ES_PM25F() { return (float) s_ES.dN_PM25F; }
float ES_CH4F () { return (float) s_ES.dN_CH4F;  }
float ES_COF  () { return (float) s_ES.dN_COF;   }
float ES_CO2F () { return (float) s_ES.dN_CO2F;  }
float ES_NOXF () { return (float) s_ES.dN_NOXF;  }
float ES_SO2F () { return (float) s_ES.dN_SO2F;  }

float ES_PM10S() { return (float) s_ES.dN_PM10S; }
float ES_PM25S() { return (float) s_ES.dN_PM25S; }
float ES_CH4S () { return (float) s_ES.dN_CH4S;  }
float ES_COS  () { return (float) s_ES.dN_COS;   }
float ES_CO2S () { return (float) s_ES.dN_CO2S;  }
float ES_NOXS () { return (float) s_ES.dN_NOXS;  }
float ES_SO2S () { return (float) s_ES.dN_SO2S;  }


float ES_PM10S_Duff() { return (float) s_ES.dN_PM10S_Duff; }
float ES_PM25S_Duff() { return (float) s_ES.dN_PM25S_Duff; }
float ES_CH4S_Duff () { return (float) s_ES.dN_CH4S_Duff;  }
float ES_COS_Duff  () { return (float) s_ES.dN_COS_Duff;   }
float ES_CO2S_Duff () { return (float) s_ES.dN_CO2S_Duff;  }
float ES_NOXS_Duff () { return (float) s_ES.dN_NOXS_Duff;  }
float ES_SO2S_Duff () { return (float) s_ES.dN_SO2S_Duff;  }



float ES_FlaDur () { return (float)s_ES.d_FlaDur; }
float ES_SmoDur () { return (float)s_ES.d_SmoDur; }

float ES_FlaCon () { return (float)s_ES.dN_FlaCon; }
float ES_SmoCon () { return (float)s_ES.dN_SmoCon; }

/**************************************************************
* Name: ES_GetEmi
* Desc: Get the emission factors that Burnup used
* NOTE: Burnup uses this factors to get Grams Per Square Meter
***************************************************************/
void ES_GetEmi (char cr[], double *flame, double *smolder)
{
  *flame = 0; 
  *smolder = 0; 
  if ( !xstrcmpi (cr,"pm10")) { *flame = d_pm10f; *smolder = d_pm10s; }  
 
/* Caller could use either  */
 if ( !xstrcmpi (cr,"pm25")) { *flame = d_pm25f; *smolder = d_pm25s; }
 if ( !xstrcmpi (cr,"pm2.5")) { *flame = d_pm25f; *smolder = d_pm25s; } 

  if ( !xstrcmpi (cr,"ch4" )) { *flame = d_ch4f;  *smolder = d_ch4s; }
  if ( !xstrcmpi (cr,"co"  )) { *flame = d_cof;   *smolder = d_cos;  }
  if ( !xstrcmpi (cr,"co2" )) { *flame = d_co2f;  *smolder = d_co2s; }
  if ( !xstrcmpi (cr,"nox" )) { *flame = d_noxf;  *smolder = d_noxs; }
  if ( !xstrcmpi (cr,"so2" )) { *flame = d_so2f;  *smolder = d_so2s; }
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: ES_Init
* Desc: Init all vars in struct to 0 in Emission Structure
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
void ES_Init (d_ES *a_ES)
{
  a_ES->d_FlaCon = 0;                     /* Store consumed for time step      */
  a_ES->d_SmoCon = 0;

  a_ES->dN_PM25F = 0;
  a_ES->dN_CH4F = 0;
  a_ES->dN_COF = 0;
  a_ES->dN_CO2F = 0;
  a_ES->dN_PM10F = 0;
  a_ES->dN_NOXF = 0;
  a_ES->dN_SO2F = 0;

  a_ES->dN_PM25S = 0;
  a_ES->dN_CH4S = 0;
  a_ES->dN_COS = 0;
  a_ES->dN_CO2S = 0;
  a_ES->dN_PM10S = 0;
  a_ES->dN_NOXS = 0;
  a_ES->dN_SO2S = 0;

  a_ES->dN_PM25S_Duff = 0;
  a_ES->dN_CH4S_Duff = 0;
  a_ES->dN_COS_Duff = 0;
  a_ES->dN_CO2S_Duff = 0;
  a_ES->dN_PM10S_Duff = 0;
  a_ES->dN_NOXS_Duff = 0;
  a_ES->dN_SO2S_Duff = 0;

  a_ES->dN_Cnt = 0; 

  a_ES->d_FlaDur = 0;
  a_ES->d_SmoDur = 0;

  a_ES->dN_FlaCon = 0;            /* Total Flame Consumed */
  a_ES->dN_SmoCon = 0;            /* Total Smolder Consumed */
  a_ES->d_DuffCon = 0; 
  a_ES->d_HSFBCon = 0; 

  a_ES->dN_FlaCon = 0;
  a_ES->dN_SmoCon = 0;
  a_ES->dN_DuffCon = 0; 
  a_ES->dN_HSFBCon = 0; 


  a_ES->d_rm_Lit = 0;
  a_ES->d_rm_1Hr  = 0;
  a_ES->d_rm_10Hr = 0;
  a_ES->d_rm_100Hr = 0;
  a_ES->d_rm_1kHr = 0;

  a_ES->d_cn_Lit = 0;
  a_ES->d_cn_1Hr  = 0;
  a_ES->d_cn_10Hr = 0;
  a_ES->d_cn_100Hr = 0;
  a_ES->d_cn_1kHr = 0;

  a_ES->d_cn_LitSmoPerCent = 0;
  a_ES->d_cn_1HrSmoPerCent = 0;
  a_ES->d_cn_10HrSmoPerCent = 0;
  a_ES->d_cn_100HrSmoPerCent = 0;   
  a_ES->d_cn_1kSmoPerCent = 0;  


  a_ES->d_cn_1kHrFla = 0; 
  a_ES->d_cn_1kHrSmo = 0; 

}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: EFM_Open
* Desc: Open the output Emissions file.
*
* Note-1: This is for the FRAMES Web Fofem application which is built
*         to run with no interface and runs using stdin and stdout
*         Sending in the e_FrameFN file name will cause the emissions
*         to go out the stdout using printf's rather than to a file.
*         This was an easy way to get the emissions out with out trying
*         to save them, REMEMBER theses emission values are generated
*         as burnup is doing its time steps.
*         Also See: EFM_Write ()   EFM_Close()
*
*   In: cr_FN.....Path File Name
*  Ret: 1 OK      0 Error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
FILE *fh_EFM;
int   EFM_Open (char cr_FN[])
{
   gfN_FlaCon = gfN_SmoCon = 0; 

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* See Note-1 above                                                          */
  if ( !strcmp (cr_FN,e_FrameFN) ) {
    fh_EFM =  e_Framefh;
    printf ("%s\n",e_Frame_Start );
    return 1; }
/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-*/


  if ( !strcmp (cr_FN,"") ) {           /* name is NULL, that's ok other     */
    fh_EFM = NULL;                      /* EFM func get called but will not  */
    return 1; }                         /* output anyting                    */

/* Larry ................................................*/
/* an '@' on the front of the file name signals that we are to   */
/*  output and emission file with flaming AND smolder values */
/*  I implemented this to accomondate the ARB (calif air resource board) */
  strcpy (gcr_EFFS, "No");

  if ( cr_FN[0] == '@' ) {
    strcpy (gcr_EFFS, "Yes");
    fh_EFM = fopen (&cr_FN[1],"w");
    if ( fh_EFM == NULL )
       return 0;

    Heat_HeadingFS (fh_EFM);                /* putting head in output report     */
    return 1;
   }


  fh_EFM = fopen (cr_FN,"w");
  if ( fh_EFM == NULL )
     return 0;

  Heat_Heading (fh_EFM);                /* putting head in output report     */
  return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: EFM_Close
* Desc: Close the Emission Ouput FIle
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
void EFM_Close()
{

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.- */
/* For Frames Web Application - see notes in EFM_Open()                      */
  if ( fh_EFM == e_Framefh  ) {
    printf ("%s\n", e_Frame_End );
    return ; }
/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.- */
float f,s;
  if ( fh_EFM != NULL ) {                     /* if file was opened           */
 //   fprintf (fh_EFM, "Flame:  %f      TPA\n", gfN_FlaCon); 
 //   fprintf (fh_EFM, "Smold:  %f      TPA:\n", gfN_SmoCon); 
    fclose (fh_EFM); }
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: EFM_Write
* Desc: Write a single line to the Emissions Output File
*       One line corresponds to a single time step.
*   In: a_ES.......Emisson struct, this has all the values calculated
*                   and ready to go.
*       d_Time.....the time line in seconds
*       d_FirInt...Fire Intensity
*  Ret: 1 OK      0 Error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int   EFM_Write (d_ES  *a_ES, double d_Time, double d_FirInt)
{
int i_Time;
double d; 

float Smo, Fla, SmoWL, FlaWL, Smo1k, Fla1k;
float Smo_tpa, Fla_tpa;

//   ES_FlaSmo (a_ES, &SmoWL, &FlaWL, &Smo1k, &Fla1k);
//   Smo = SmoWL + Smo1k;
//   Fla = FlaWL + Fla1k;
//   Fla_tpa = KgSq_To_TPA(Fla);
//   Smo_tpa = KgSq_To_TPA (Smo); 


// test...................................
//   gfN_FlaCon += Fla_tpa; 
//   gfN_SmoCon += Smo_tpa; 


// test...................................

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* For Frames Web Application - see notes in EFM_Open()                      */
  if ( fh_EFM == e_Framefh  ) {
   i_Time = (int) d_Time;
   printf ("%4d  %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f\n",
     i_Time,
     d_FirInt,
     a_ES->d_PM25,
     a_ES->d_PM10,
     a_ES->d_CH4,
     a_ES->d_CO2,
     a_ES->d_CO,
     a_ES->d_NOX,
     a_ES->d_SO2,
     a_ES->d_FlaCon,
     a_ES->d_SmoCon);
    return 1; }
/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/

  if ( fh_EFM == NULL )                      /* if file was opened           */
     return 1;

  i_Time = (int) d_Time;

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Do the emission file with individual flaming and smoldering values */
  if ( !strcmp (gcr_EFFS, "Yes")) {
    fprintf (fh_EFM, "%4d  %9.4f  %9.4f %9.4f   %9.4f %9.4f    %9.4f %9.4f    %9.4f %9.4f    %9.4f %9.4f    %9.4f %9.4f    %9.4f %9.4f    %9.4f %9.4f\n",
       i_Time,
       d_FirInt,

       a_ES->d_PM25F,    a_ES->d_PM25S,
       a_ES->d_PM10F,    a_ES->d_PM10S,
       a_ES->d_CH4F,     a_ES->d_CH4S,
       a_ES->d_CO2F,     a_ES->d_CO2S,
       a_ES->d_COF,      a_ES->d_COS,

       a_ES->d_NOXF,     a_ES->d_NOXS,
       a_ES->d_SO2F,     a_ES->d_SO2S,

       a_ES->d_FlaCon,
       a_ES->d_SmoCon);
    return 1;
  }



/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Regular emission out */


  fprintf (fh_EFM, "%4d  %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f  %9.4f %9.4f\n",
     i_Time,
     d_FirInt,
     a_ES->d_PM25,
     a_ES->d_PM10,
     a_ES->d_CH4,
     a_ES->d_CO2,
     a_ES->d_CO,

     a_ES->d_NOX,
     a_ES->d_SO2,

     a_ES->d_FlaCon,
     a_ES->d_SmoCon,

     KgSq_To_TPA( a_ES->d_FlaCon),
     KgSq_To_TPA ( a_ES->d_SmoCon) ); 


    gfN_FlaCon += KgSq_To_TPA( a_ES->d_FlaCon);
    gfN_SmoCon += KgSq_To_TPA ( a_ES->d_SmoCon); 



  return 1;
}

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.- */
double pow2(double input)
{
  return input*input;
}

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.- */
double func(double h, double theta)
{
const double a = 0.7478556;
const double b = 0.4653628;
const double c = 0.1282064;
   return h*(b-h*(c-h))-(1.0-theta)/a;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: BRN_Intensity
* Desc: Calculate Intensity for Herb, Shrub, Foliga and Branch,it
*       will be added to first line in Emissions File
*   In: f_Con....Consumed amount (herb+shrub+foliage+branch) tons per acre 
*  Ret: intensity - see following comments for units 
*
* 4-13-15 - Got the following notes from DL 
*  A = heat content BTU/lb
*  B = convert tons to lb
*  C = convert ft2 to acre
*  D = convert min to sec
*  E = convert BTU/ft2/sec to kW/m2
*  [[(a * b) * e]/(c * d)]
* NOTE - I tested to make sure the above equation produces the same answer 
*        a the was it's done below, which doesn't have all the braces 
*
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
float BRN_Intensity (float f_Con, float f_Seconds)
{
float a,b,c,d,e,f,x;
   a = 8000;
   b = 2000;
   c = 43560;

 //  d = 60;
   d = f_Seconds; 

   e = 11.35;
   f = a * b / c / d * e;
   f = f * f_Con;
   return f;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Save_SGV
* Desc: Save the Emissions for the time step to the SGV, which is
*        a struct used to graph the emissions
*        The structure will have info about each time step.
*   In: a_ES......Emission Struct
*       d_time....the time (seconds)
*       d_FirInt..Fire Intensity
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
void Save_SGV (d_ES *a_ES, double d_time, double d_FirInt)
{
d_SGV s_SGV;
   s_SGV.f_Sec    = (float) d_time;          /* saving values at each time   */
   s_SGV.f_PM2_5  = (float) a_ES->d_PM25;    /* step, for graphing later     */
   s_SGV.f_PM10   = (float) a_ES->d_PM10;
   s_SGV.f_CH4    = (float) a_ES->d_CH4;
   s_SGV.f_CO2    = (float) a_ES->d_CO2;
   s_SGV.f_CO     = (float) a_ES->d_CO;

   s_SGV.f_NOX    = (float) a_ES->d_NOX;
   s_SGV.f_SO2    = (float) a_ES->d_SO2;

   s_SGV.f_Inten  = (float) d_FirInt;

   s_SGV.f_FlaCon = (float) a_ES->d_FlaCon;
   s_SGV.f_SmoCon = (float) a_ES->d_SmoCon;
   s_SGV.f_DuffCon   = (float) a_ES->d_DuffCon; 
   s_SGV.f_HSFBCon = (float) a_ES->d_HSFBCon; 

   s_SGV.f_rm_Lit   = (float) a_ES->d_rm_Lit;
   s_SGV.f_rm_1Hr   = (float) a_ES->d_rm_1Hr;
   s_SGV.f_rm_10Hr  = (float) a_ES->d_rm_10Hr;
   s_SGV.f_rm_100Hr = (float) a_ES->d_rm_100Hr;
   s_SGV.f_rm_1kHr  = (float) a_ES->d_rm_1kHr;

   s_SGV.f_cn_Lit   = (float) a_ES->d_cn_Lit;
   s_SGV.f_cn_1Hr   = (float) a_ES->d_cn_1Hr;
   s_SGV.f_cn_10Hr  = (float) a_ES->d_cn_10Hr;
   s_SGV.f_cn_100Hr = (float) a_ES->d_cn_100Hr;
   s_SGV.f_cn_1kHr  = (float) a_ES->d_cn_1kHr;

   s_SGV.f_cn_1kHrFla = (float) a_ES->d_cn_1kHrFla; 
   s_SGV.f_cn_1kHrSmo = (float) a_ES->d_cn_1kHrSmo; 

   s_SGV.f_pcSmo = (float) a_ES->d_pcSmo; 

/* percent of the above Consumed Fuel for this time step that was smodering */
/* Lit, 1Hr, 10Hr, 100Hr  will always be 100 percent smolder or 0 for flame */
/* 1kHr can be 0 -> 100 because 1kHr has sub compoents 3,6,9,20 - Snd * Rot */
/*   so some may go up in flame and other in smolder  */
   s_SGV.f_cn_LitSmoPerCent   = (float) a_ES->d_cn_LitSmoPerCent;
   s_SGV.f_cn_1HrSmoPerCent   = (float) a_ES->d_cn_1HrSmoPerCent;
   s_SGV.f_cn_10HrSmoPerCent  = (float) a_ES->d_cn_10HrSmoPerCent;
   s_SGV.f_cn_100HrSmoPerCent = (float) a_ES->d_cn_100HrSmoPerCent;
   s_SGV.f_cn_1kSmoPerCent    = (float) a_ES->d_cn_1kSmoPerCent;   



   SGV_Set ( &s_SGV );                       /*save 'em                      */
}



/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Bur_ChkArgs
* Desc: Check that each argument is a number
*  Ret: 0 OK, else number of argument in error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int   Bur_ChkArgs (char cr_Line[])
{
char cr1[50], cr2[50], cr3[50],  cr4[50], cr5[50],  cr6[50];
char cr7[50], cr8[50], cr9[50], cr10[50], cr11[50], cr12[50];

   cr1[0] = cr2[0] = cr3[0] = cr4[0]  = cr5[0]  = cr6[0] = 0;
   cr7[0] = cr8[0] = cr9[0] = cr10[0] = cr11[0] = cr12[0] = 0;

   sscanf (cr_Line, "%s %s %s %s %s %s %s %s %s %s %s %s\n",
         cr1, cr2, cr3, cr4, cr5, cr6, cr7, cr8, cr9, cr10, cr11, cr12);

   if ( Get_NumTyp (cr1)  == 'X' ) return 1;
   if ( Get_NumTyp (cr2)  == 'X' ) return 2;
   if ( Get_NumTyp (cr3)  == 'X' ) return 3;
   if ( Get_NumTyp (cr4)  == 'X' ) return 4;
   if ( Get_NumTyp (cr5)  == 'X' ) return 5;
   if ( Get_NumTyp (cr6)  == 'X' ) return 6;
   if ( Get_NumTyp (cr7)  == 'X' ) return 7;
   if ( Get_NumTyp (cr8)  == 'X' ) return 8;
   if ( Get_NumTyp (cr9)  == 'X' ) return 9;
   if ( Get_NumTyp (cr10) == 'X' ) return 10;
   if ( Get_NumTyp (cr11) == 'X' ) return 11;
   if ( Get_NumTyp (cr12) == 'X' ) return 12;
   return 0;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name:
* Desc:
*  Ret:
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
void Bur_Error (char cr_Err[], char cr1[], char cr_Line[])
{

  strcpy (cr_Err, cr1);
  if ( strcmp (cr_Line,"") ) {
    strcat (cr_Err,"\n Line in Error:\n");
    if ( strlen (cr_Line) > 40 )        /* if line is too long it will trash */
      cr_Line[39] = 0;                  /* the strings its going to          */
    strcat (cr_Err,cr_Line); }

}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: BRN_ChkSwi
* Desc: Checks to see if string is a valid switch
*  Ret:
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int  BRN_ChkSwi (char cr[])
{
   if (_stristr(cr, e_swMAX_TIMES    ))
     return 1;
   if (_stristr(cr, e_swINTENSITY    )) 
     return 1;
   if (_stristr(cr, e_swIG_TIME      ))
     return 1;
   if (_stristr(cr, e_swWINDSPEED    )) 
     return 1;
   if (_stristr(cr, e_swDEPTH        )) 
     return 1;
   if (_stristr(cr, e_swAMBIENT_TEMP )) 
     return 1;
   if (_stristr(cr, e_swR0           )) 
     return 1;
   if (_stristr(cr, e_swDR           )) 
     return 1;
   if (_stristr(cr, e_swTIMESTEP     ))
     return 1;
   if (_stristr(cr, e_swDUFF_LOAD    ))
     return 1;
   if (_stristr(cr, e_swDUFF_MOIST   ))
     return 1;
   
   return 0;
}


/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: CFF_ChkAll
* Desc: See if all of the switches have been found yet.
*       Whenever a switch is found it gets a 1
*  Ret: 1 All switch have been found, else 0
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int   CFF_ChkAll (d_CFF  *a_CFF)
{
   if ( a_CFF->i_MAX_TIMES    == 0 ) return 0;
   if ( a_CFF->i_INTENSITY    == 0 ) return 0;
   if ( a_CFF->i_IG_TIME      == 0 ) return 0;
   if ( a_CFF->i_WINDSPEED    == 0 ) return 0;
   if ( a_CFF->i_DEPTH        == 0 ) return 0;
   if ( a_CFF->i_AMBIENT_TEMP == 0 ) return 0;
   if ( a_CFF->i_r0           == 0 ) return 0;
   if ( a_CFF->i_dr           == 0 ) return 0;
   if ( a_CFF->i_TIMESTEP     == 0 ) return 0;
   if ( a_CFF->i_DUFF_LOAD    == 0 ) return 0;
   if ( a_CFF->i_DUFF_MOIST   == 0 ) return 0;
   return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: CFF_Init
* Desc: init all fields to 0
* In/Out: a_CFF
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
void CFF_Init (d_CFF *a_CFF)
{
   a_CFF->i_MAX_TIMES   = 0;
   a_CFF->i_INTENSITY   = 0;
   a_CFF->i_IG_TIME     = 0;
   a_CFF->i_WINDSPEED   = 0;
   a_CFF->i_DEPTH       = 0;
   a_CFF->i_AMBIENT_TEMP= 0;
   a_CFF->i_r0          = 0;
   a_CFF->i_dr          = 0;
   a_CFF->i_TIMESTEP    = 0;
   a_CFF->i_DUFF_LOAD   = 0;
   a_CFF->i_DUFF_MOIST  = 0;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: bstrupr
* Desc: convert a string to upper case, strupr is not on Unix
* In/Out:
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
void  bstrupr (char cr[])
{
int i;
char c;
  for ( i = 0; i < 20000; i++ ) {
    if ( cr[i] == 0 )
      break;
    c = (char) toupper (cr[i]);
    cr[i] = c; }
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: BRN_SetFuel
* Desc: set Fuel Load associtated values in to burnup variable arrays.
*       This is used to set some Burnup input variable when a Burnup input
*        file is not being used.
*       THis basicaly does what is equivalent to a single fuel load line
*        of a burnup input file.
* Note-1: Multiply by 1000 because that's what the orginal burnup code did
*          when it would read in the value from input file
*        cr_SR.....Sound or Rotten, "SND", "ROT"
*        f_Load....fuel load
*        f_Moist...moistuer
*        f_Sigma...see defines caller is using
*  In/Out: aiX....index where to put stuff into burnup arrays
*                   this index will get incremented after use
*  Out: cr_ErrMes...NULL or Error Message when one occurs
*  Ret: 1 OK data was set,
*       0 no load was sent in, that's ok we just don't use it, but let
*          caller
*       -1 Error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int BRN_SetFuel (int *aiX, char cr_SR[], float f_Load, float f_Moist, float f_Sigma)
{
int iX;
   if ( f_Load <= 0 )                        /* no load, so don't do anything*/
     return 0;
   iX = *aiX;                                /* where in array to put stuff  */
   if ( iX >= MAXNO ) {                      /* Don't exceed array size      */
     LEM_Put ("BRN_SetFuel", "fuel loads exceeds burnup array size");
     iX = MAXNO -1; }                        /* See Note-1 above             */

   if ( !strcmp (cr_SR,"SND") ) {            /* Sound Wood                   */
      dendry[iX] = e_Snd_dendry;
      tpig  [iX] = e_Snd_tpig + e_tpig_adj;}
   else if ( !strcmp (cr_SR,"ROT") ) {       /* Rotten Wood                  */
      dendry[iX] = e_Rot_dendry;
      tpig  [iX] = e_Rot_tpig + e_tpig_adj;}
   else {
      LEM_Put ("BRN_SetFuel", "Internal Logic, Fuel Load type unknown, will assign 'Sound'");
      dendry[iX] = e_Snd_dendry;
      tpig  [iX] = e_Snd_tpig + e_tpig_adj;}

   wdry  [iX] = f_Load;                      /* fuel load                    */
   htval [iX] = e_htval * 1000;              /* See Note-1 above             */
   fmois [iX] = f_Moist;
   sigma [iX] = f_Sigma;
   cheat [iX] = e_cheat;
   condry[iX] = e_condry;
   tchar [iX] = e_tchar + e_tchar_adj;
   ash   [iX] = e_ash;
   number++;                                /* tells number of loads         */
   iX++;                                    /* up index                     */
   *aiX = iX;                               /* and send it back             */
   return 1;
}

/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: Min & Max
* Desc: these are not ANSI standard, compile gets an error so need to
*       make my own.
*   In:
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{**/
double  Min (double a, double b)
{
  if ( a < b )
    return a;
  return b;
}

double  Max (double a, double b)
{
  if ( a > b )
    return a;
  return b;
}

#ifdef wowowowow
/*{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}
* Name: BRN_ReadInFil
* Desc: Read the input burnup input file.
* Note: The design of the input file was not of my making so gotta deal
*         with it as is. It has those 11 switchs on the front then a file
*         name (that doesn't appear to have any reason for being there)
*         then any number of lines with the fuel load info.
*         So I:
*          - get the 11 switches at the top of the file.
*          - look for that bogus file name and get past it
*             I wrote code so if it's not there that's ok
*          - get the fuel loading lines
*   In: cr_FN......File Name
*  Out: cr_Err.....Error Message
*  Ret: 1 OK,    0 Error
{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}{*}*/
int BRN_ReadInFil (char cr_FN[], char cr_Err[])
{
int    i, i_Ret, iN;
long   v, w;
#define eC_Line 250
char   cr_Line[eC_Line], cr1[100], cr2[100], *eof, c_Type;
double d_a, d_b, d_c, d_d, d_e, d_f, d_g, d_h, d_x, d_y;
FILE   *fh_In;
d_CFF  s_CFF;

long    l_ntimes = e_MAX_TIMES;/* max number of iterations                   */
long    l_NumFuelCats = 8;     /* number of fuel categories this time        */
double  d_fi = 150.0;          /* initializing fire intensity (kW/m2/s)      */
double  d_ti = 30.0;           /* duration of initial intensity pulse  (s)   */
double  d_WinSpe = 0.0;
double  d_FueDep = 0.523;      /* fuelbed depth (m)                          */
double  d_tamb = 27.0;         /* ambient temperature (C)                    */
double  d_r0   = 1.8;          /* minimum value of mixing parameter          */
double  d_dr   = 0.4;          /* maximum addition to mixing parameter       */
double  d_dt   = 10;           /* timestep                                   */
double  d_wdf  = 11.0;         /* duff weight (kg/m2)                        */
double  d_dfm  = 1.62;         /* duff moisture content (fraction)           */


   strcpy (cr_Err,"");
   fh_In = fopen (cr_FN, "r");               /* Open Input File              */
   if (  fh_In == NULL) {
      sprintf(cr_Err,"Can't Open Burnup Tmp Input file: %s \n", cr_FN);
      return 0; }

   CFF_Init (&s_CFF);                        /* to check for switches        */
   iN = 0;

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Get the 11 switches/args at top of file                                   */
   i_Ret = 0;
   while ( 1 ) {
     if ( CFF_ChkAll (&s_CFF) )              /* see if have all switches yet */
       break;                                /*  OK gottem' all              */
     eof = fgets ( cr_Line, 200, fh_In  );   /* Read a line from file        */
     if ( eof == NULL )  {                   /* shouldn't hit EOF            */
       Bur_Error (cr_Err,"Missing Switch(s)","");
       i_Ret = 0;
       goto X; }

     Blk_End_Line (cr_Line,eC_Line);         /* get rid of end line chars    */
     if ( isBlank(cr_Line) )                 /* skip blank lines             */
       continue;
     strcpy (cr1,"");
     strcpy (cr2,"");
     sscanf (cr_Line, "%s %s", cr1,cr2);     /* get switch & argument        */
     if ( !strcmp (cr1,"") )                 /* Empty line                   */
        continue;
     if ( cr1[0] == '#' )                    /* Skip any comments lines      */
       continue;

     bstrupr (cr1);                          /* get string to upper case     */

     if ( !BRN_ChkSwi (cr1) ) {              /* check for valid switch       */
        Bur_Error (cr_Err,"Invalid Switch.\n",cr_Line);
        i_Ret = 0;
        goto X; }

     if ( !strcmp (cr2,"") ) {               /* must have an arg             */
        Bur_Error (cr_Err,"Switch Has Missing Argument.\n",cr_Line);
        i_Ret = 0;
        goto X; }

     c_Type = Get_NumTyp (cr2);              /* 2cd arg must be number       */
     if ( c_Type == 'X' ) {
        Bur_Error (cr_Err,"Invalid Argument.\n",cr_Line);
        i_Ret = 0;
        goto X; }

     if (!strcmp(cr1, e_swMAX_TIMES )) {
          l_ntimes = atoi(cr2);
          s_CFF.i_MAX_TIMES++;
          continue; }
     if (strstr(cr1, e_swINTENSITY)) {
         d_fi = atof(cr2);
         s_CFF.i_INTENSITY++;
         continue;}
     if (strstr(cr1, e_swIG_TIME      )){ d_ti     = atof(cr2); s_CFF.i_IG_TIME++;      continue;}
     if (strstr(cr1, e_swWINDSPEED    )){ d_WinSpe = atof(cr2); s_CFF.i_WINDSPEED++;    continue;}
     if (strstr(cr1, e_swDEPTH        )){ d_FueDep = atof(cr2); s_CFF.i_DEPTH++;        continue;}
     if (strstr(cr1, e_swAMBIENT_TEMP )){ d_tamb   = atof(cr2); s_CFF.i_AMBIENT_TEMP++; continue;}
     if (strstr(cr1, e_swR0           )){ d_r0     = atof(cr2); s_CFF.i_r0++;           continue; }
     if (strstr(cr1, e_swDR           )){ d_dr     = atof(cr2); s_CFF.i_dr++;           continue; }
     if (strstr(cr1, e_swTIMESTEP     )){ d_dt     = atof(cr2); s_CFF.i_TIMESTEP++;     continue; }
     if (strstr(cr1, e_swDUFF_LOAD    )){ d_wdf    = atof(cr2); s_CFF.i_DUFF_LOAD++;    continue; }
     if (strstr(cr1, e_swDUFF_MOIST   )){ d_dfm    = atof(cr2); s_CFF.i_DUFF_MOIST++;   continue;}
     Bur_Error(cr_Err,"Invalid Switch",cr_Line);
     i_Ret = 0;
     goto X;  }

     BRN_SetFireDat (l_ntimes,d_fi,d_ti,d_WinSpe,
                     d_FueDep,d_tamb,d_r0,d_dr,d_dt,d_wdf,d_dfm);

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* Look for bogus file name, that's in the file                              */
   while ( 1 ) {
     eof = fgets ( cr_Line, 200, fh_In  );   /* Read a line from file        */
     if ( eof == NULL ) {                    /* End of File                  */
       Bur_Error(cr_Err,"No fuel loading data found in file.","");
       i_Ret = 0;
       goto X; }

     Blk_End_Line (cr_Line,eC_Line);         /* get rid of end line chars    */
     if ( isBlank(cr_Line) )                 /* skip blank lines             */
       continue;

     strcpy (cr1,"");
     sscanf (cr_Line, "%s", cr1);
     if ( cr1[0] == '#' )                    /* Skip any comments lines      */
       continue;
     if ( !strcmp (cr1,"") )                 /* skip empty lines             */
        continue;

     c_Type = Get_NumTyp (cr1);              /* get number type              */
     if ( c_Type == 'I' )                    /* integer                      */
       goto Num;                             /* go do fuel loading line      */
     if ( c_Type == 'F' )                    /* float                        */
       goto Num;                             /* go do fuel loading line      */
     if ( c_Type == 'X' )                    /* must be the bogus file name  */
       break;                                /* time to go do fuel lines     */
     strcpy (cr_Err,"Logic Error - BRN_ReadInFil");
     i_Ret = 0;
     goto X;
   }

/*.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.*/
/* read in the fuel loading lines                                            */
   while ( 1 ) {
     eof = fgets ( cr_Line, 200, fh_In  );   /* Read a line from file        */
     if ( eof == NULL )                      /* End of File                  */
       break;

     Blk_End_Line (cr_Line,eC_Line);         /* get rid of end line chars    */
     if ( isBlank(cr_Line) )                 /* skip blank lines             */
       continue;

     strcpy (cr1,"");
     sscanf (cr_Line, "%s", cr1);
     if ( cr1[0] == '#' )                    /* Skip any comments lines      */
       continue;
     if ( !strcmp (cr1,"") )
       continue;                             /* skip empty lines             */

Num:
     i = Bur_ChkArgs (cr_Line);              /* make sure they're all nums   */
     if ( i > 0 ) {
       sprintf (cr1,"argument %d is not numberic or there is not 12 numbers on line.\n",i);
       Bur_Error (cr_Err, cr1,cr_Line);
       i_Ret = 0;
       goto X; }

     sscanf (cr_Line, "%ld %ld %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
              &v,&w,&d_a,&d_b,&d_c,&d_d,&d_e,&d_f,&d_g,&d_h,&d_x,&d_y);

     if ( d_a <= 0.0)                   /* if no fuel Load                   */
        continue;
     wdry  [iN] = d_a;                  /* load into arrays that burnup uses */
     htval [iN] = d_b*1000.0;           /*  to do its calcs                  */
     fmois [iN] = d_c;
     dendry[iN] = d_d;
     sigma [iN] = d_e;
     cheat [iN] = d_f;
     condry[iN] = d_g;
     tpig  [iN] = d_h + e_tpig_adj;
     tchar [iN] = d_x + e_tchar_adj;
     ash   [iN] = d_y;
     iN++;
     number = iN;                       /* number of fuel loads in arrays    */

     if ( iN == MAXNO ) {
       Bur_Error (cr_Err, "Too Many fuel loads in burnup input file.\n","");
       i_Ret = 0;
       goto X; }
   }

   if ( iN == 0 ) {
     Bur_Error (cr_Err, "No fuel load data found in input file \n","");
     i_Ret = 0;
     goto X; }

   if ( !BRN_CheckData(cr_Err) ) {           /* Check input data to burnup   */
     i_Ret = 0;
     goto X; }                               /* Error                        */

   i_Ret = 1;                                /* OK no errors                 */
X:
   fclose (fh_In);
   return i_Ret;
}

#endif 

/******************************************************************
* Name: BRN_SetEmis
* Desc: Set the emission factors and the critical fire intensity
*       point, which is used to determine if flame or smolder
*       factors get used.
* NOTE: the f, s and d prefixes in the variable names
*       f = flaming, d = duff, s = smoldering coarse woody
*******************************************************************/
void BRN_SetEmis (float f_CriInt,
   float f_fCO, float f_fCO2, float f_fCH4, float f_fPM25, float f_fPM10, float f_fNOX, float f_fSO2,
   float f_sCO, float f_sCO2, float f_sCH4, float f_sPM25, float f_sPM10, float f_sNOX, float f_sSO2,
   float f_dCO, float f_dCO2, float f_dCH4, float f_dPM25, float f_dPM10, float f_dNOX, float f_dSO2)
{

    gf_CriInt = f_CriInt;

    gf_fCO   =  f_fCO;
    gf_fCO2  =  f_fCO2;
    gf_fCH4  =  f_fCH4;
    gf_fPM25 =  f_fPM25;
    gf_fPM10 =  f_fPM10;
    gf_fNOX  =  f_fNOX;
    gf_fSO2  =  f_fSO2;

    gf_sCO   =  f_sCO;
    gf_sCO2  =  f_sCO2;
    gf_sCH4  =  f_sCH4;
    gf_sPM25 =  f_sPM25;
    gf_sPM10 =  f_sPM10;
    gf_sNOX  =  f_sNOX;
    gf_sSO2  =  f_sSO2;

    gf_dCO   =  f_dCO;
    gf_dCO2  =  f_dCO2;
    gf_dCH4  =  f_dCH4;
    gf_dPM25 =  f_dPM25;
    gf_dPM10 =  f_dPM10;
    gf_dNOX  =  f_dNOX;
    gf_dSO2  =  f_dSO2;
}

/**************************************************************
* Name: ES_SetComponents
* Desc: Return the amount of 1000k ( 3+ inch) that is consumed
*        for each time step.
*       This is done by getting and saving the remaining fuel
*        loads at everytime step and take the difference from
*        the current remaining fuel loads.
* Note-1: fuel compenent stuff is kept in a big arrary for
*         the simulation, there seems to be more than one
*         entry for a component, and the loc(x,x) functions 
*         determines where they are.
* Note-2: We determine what is 3+ wood by looking at the the sigma
*         array which is the SAV Surface Area to Volume ratio, 
*         the entries correspond to the fuel load array
*         I need to add a slight amount to the 3+ SAV to get it
*         to do the compare right, probaby due to it being a 
*         double to float compare or something like that
* Note-3: Figure out the burn out time by taking diff of last remaining 
*         load and current that gives us a consumed amount, then we keep 
*         saving the current time as long as there is a consumed amount, 
*         so the time represents the last time we had something consumed, 
*
* Things to keep in mind:
*       burnup simimulation itself only does lit,1,10,100 and 1000k fuels
*        so we don't deal with duff, herb, shrb, foli branch here
*       the only fuel components we'll see here are the ones loaded
*        into burnup, so if there was no 100 Hr fuels inputed than
*        we won't see any here.
*   In: iS_Init.....1 to init, this must be done before any
*                   burnup time step is done
*  Out: af_TPA......consumed amount in tons per acre 
*                   This is done for test
*  Ret: the consumed amount in burnup units, Kg Sq m
*************************************************************/
double  dr_Lst[MAXNO];
double  dr_Cur[MAXNO];
double  dr_Dif[MAXNO];
double  gf_TotConTPA;

/********************************************************************
*  see notes above 
*
*********************************************************************/
double  ES_SetComponents (int iS_Init, float *af_TPA, double d_Time, d_ES *a_ES)
{
int i, m,n, mn;
float f;
double  rem, d_Coarse, d_Con; 
double d_SAV;
double s, ti, ts, to, tf, wd;

   if ( iS_Init == 1 ) {
     gf_TotConTPA = 0 ;
     for ( i = 0; i < MAXNO; i++ ) 
       dr_Lst[i] = dr_Cur[i] = dr_Dif[i] = dr_Done[i] = dr_DD[i] = 0;  
   }
   rem = 0; 

/*......................................................................*/
/* This loop code was taken from the Summary() function in this file */
   for ( m = 1; m <= number; m++) {     /* for each fuel componet */
     rem = 0.0;
     for ( n = 0; n <= m; n++) {       /* get each entry for a component */
       mn = loc(m, n);                 /* Note-1 above */
       wd = WOO[mn];                    /* Get the remaining fuel load */
       rem = rem + wd; 
	 }
     dr_Cur[m-1] = rem;                /* save the amount for each component */
   }   /*  for m */ 
/*..................................................................*/

   if ( iS_Init == 1 ) {
     for ( i = 0; i < number; i++ ) 
       dr_Lst[i] = dr_Cur[i];
     return 0;  }

/* save burn out time for each fuel component, See Note-3 above */
   for ( i = 0; i < number; i++ ) {
     d_Con = dr_Lst[i] - dr_Cur[i]; 
     if ( d_Con > 0 )                        //  0.0000001 )   
       dr_Done[i] = d_Time ; 
   } /* for i */ 

   for ( i = 0; i < number; i++ )       /* Take difference since last */
     dr_Dif[i] = dr_Lst[i] - dr_Cur[i]; /* to get the amount consumed on this time step*/

/* Load ES with individual fuel component consumed fuel amts and percent smolder */
   for ( i = 0; i < number; i++ )
     ES_SetComp (i, dr_Lst[i], dr_Dif[i], a_ES );
   
 /* total wood lit consumed in smoldering  */
   a_ES->d_cn_SmoConPC = FIC_Percent (); 

   for ( i = 0; i < number; i++ )       /* now save these to Last */
     dr_Lst[i] = dr_Cur[i];

   d_Coarse = 0;
   d_SAV = e_SURat_DWk_3_6 + 1.0;     /* See Note-2 above */
   for ( i = 0; i < number; i++ ) { 
     s = sigma[i];
     if ( s <= d_SAV  )
       d_Coarse += dr_Dif[i];
    }

   f = KgSq_To_TPA ((float)d_Coarse);
   *af_TPA = f;                    /* Probably don't need this but helps in testing */
   gf_TotConTPA += f;              /* Keeping this total for testing */

   return d_Coarse ;
}


/*************************************************************
* Name: _SetFuelComp
* Desc: Set the individual fuel Component values for consumed,
*       remaining fuel, and percent smolder consumed       
  e_SURat_Lit       8200.0
  e_SURat_DW1       1480.0
  e_SURat_DW10       394.0
  e_SURat_DW100      105.0
  e_SURat_DWk_3_6     39.4
  e_SURat_DWk_6_9     21.9
  e_SURat_DWk_9_20    12.7
  e_SURat_DWk_20      5.91
* Note-1:  Need to add 1.0 to make '>=' comparisions work
*          39.4 if really  39.400001 (approx.)            
*
************************************************************/
int ES_SetComp (int i, double d_Rem, double d_Con, d_ES *a_ES )
{
double  d_SURat_DW1      = e_SURat_DW1     + 1.0;
double  d_SURat_DW10     = e_SURat_DW10    + 1.0;
double  d_SURat_DW100    = e_SURat_DW100   + 1.0;
double  d_SURat_DWk_3_6  = e_SURat_DWk_3_6 + 1.0;

  if ( i == 0 ) {
    a_ES->d_rm_Lit   = 0;
    a_ES->d_rm_1Hr   = 0;
    a_ES->d_rm_10Hr  = 0;
    a_ES->d_rm_100Hr = 0;
    a_ES->d_rm_1kHr  = 0;

    a_ES->d_cn_Lit   = 0;
    a_ES->d_cn_1Hr   = 0;
    a_ES->d_cn_10Hr  = 0;
    a_ES->d_cn_100Hr = 0;
	a_ES->d_cn_1kHr  = 0; 

   a_ES->d_cn_LitSmoPerCent =  0;    
   a_ES->d_cn_1HrSmoPerCent = 0;     
   a_ES->d_cn_10HrSmoPerCent = 0;   
   a_ES->d_cn_100HrSmoPerCent = 0;   
   a_ES->d_cn_1kSmoPerCent = 0;  }

   if ( sigma[i] <= d_SURat_DWk_3_6  ) {
     a_ES->d_rm_1kHr += d_Rem;
     a_ES->d_cn_1kHr += d_Con; 
     a_ES->d_cn_1kSmoPerCent = FIC_1kSmoPercent ();  }

   else if ( sigma[i] <= d_SURat_DW100  ) {
     a_ES->d_rm_100Hr += d_Rem;
     a_ES->d_cn_100Hr += d_Con; 
     a_ES->d_cn_100HrSmoPerCent = FIC_SmoPercent (e_SURat_DW100); }

   else if ( sigma[i] <= d_SURat_DW10  ) {
     a_ES->d_rm_10Hr += d_Rem;
     a_ES->d_cn_10Hr += d_Con; 
     a_ES->d_cn_10HrSmoPerCent = FIC_SmoPercent (e_SURat_DW10);   }

   else if ( sigma[i] <= d_SURat_DW1  ) {
     a_ES->d_rm_1Hr += d_Rem;
     a_ES->d_cn_1Hr += d_Con;
     a_ES->d_cn_1HrSmoPerCent = FIC_SmoPercent (e_SURat_DW1);     }

   else {
     a_ES->d_rm_Lit += d_Rem;
     a_ES->d_cn_Lit += d_Con;
     a_ES->d_cn_LitSmoPerCent = FIC_SmoPercent (e_SURat_Lit);      }

   return 1;
}

/******************************************************************
*
  e_SURat_Lit       8200.0
  e_SURat_DW1       1480.0
  e_SURat_DW10       394.0
  e_SURat_DW100      105.0
  e_SURat_DWk_3_6     39.4
  e_SURat_DWk_6_9     21.9
  e_SURat_DWk_9_20    12.7
  e_SURat_DWk_20      5.91
*
*******************************************************************/
/* Fire Intensity Component */
typedef struct {
   double d_SIRat;    /* Which Component, Lit, 1Hr, 10hr, etc */
   double d_dendry;   /* for 1k fuels tells if sound or rotten */
   double term;
   double test;
   double wnoduff; 
   char  cr_FS[10];
 } d_FIC;
d_FIC s_FIC[] = { 
   { e_SURat_Lit,     0,   0, 0, 0,  "" },  // 8200.0
   { e_SURat_DW1,     0,   0, 0, 0,  "" },  // 1480.0
   {e_SURat_DW10,     0,   0, 0, 0,  "" },  //  394.0
   {e_SURat_DW100,    0,   0, 0, 0,  "" },  //  105.0
                         
   {e_SURat_DWk_3_6,  e_Snd_dendry,   0, 0, 0,  "" },  //   39.4
   {e_SURat_DWk_6_9,  e_Snd_dendry,   0, 0, 0,  "" },  //   21.9
   {e_SURat_DWk_9_20, e_Snd_dendry,   0, 0, 0,  "" }, //   12.7
   {e_SURat_DWk_20,   e_Snd_dendry,   0, 0, 0,  "" }, //    5.91 
                                              
   {e_SURat_DWk_3_6,   e_Rot_dendry,   0, 0, 0,  "" },  //   39.4
   {e_SURat_DWk_6_9,   e_Rot_dendry,   0, 0, 0,  "" },  //   21.9
   {e_SURat_DWk_9_20,  e_Rot_dendry,   0, 0, 0,  "" }, //   12.7
   {e_SURat_DWk_20,    e_Rot_dendry,   0, 0, 0,  "" }, //    5.91 

   {-1, 0, 0, ' ' }};


/************************************************************
*
*
*************************************************************/
double FIC_Percent ()
{
int i;
double f,g;
double Fla, Smo;

   Fla = Smo = 0 ; 
   for ( i = 0; i < 100; i++ ) {
     if (s_FIC[i].d_SIRat < 0 ) 
       break;
     if ( !strcmp (s_FIC[i].cr_FS,"S") )
        Smo += s_FIC[i].wnoduff;
     if ( !strcmp (s_FIC[i].cr_FS,"F") )
        Fla += s_FIC[i].wnoduff;
    }

   f = Smo + Fla;

   g = Smo / f; 
   return g; 
}

/************************************************************
*
*
*************************************************************/
void FIC_Init ()
{
int i;
   for ( i = 0; i < 100; i++ ) {
     if (s_FIC[i].d_SIRat < 0 ) 
       break;
     s_FIC[i].term = 0;
     s_FIC[i].test = 0;
     s_FIC[i].wnoduff = 0; 
     strcmp (s_FIC[i].cr_FS,"");
   }
}

/************************************************************
*
*
*************************************************************/
double FIC_SmoPercent (double d_SURat)
{
int i;
   for ( i = 0; i < 100; i++ ) {
     if ( _isEqual ( s_FIC[i].d_SIRat,d_SURat) ) {
       if (!strcasecmp (s_FIC[i].cr_FS, "S" ) )
          return 1.0;
       else
          return 0;  
      } 
        
   } /* for i */ 

   return 0; 
}
/********************************************************************
* Name: FIC_1kSmoPercent 
* Desc: Get the percent of 1k hour that goes up in smoldering
*
********************************************************************/
double FIC_1kSmoPercent ()
{
int i; 
double tot,smo,f; 
    tot = smo = f = 0; 
    for ( i = 0; i < 100; i++ ) {
      if ( s_FIC[i].d_SIRat < 0 ) 
         break; 
      if ( s_FIC[i].d_SIRat > ( e_SURat_DWk_3_6  + 1.0 ) )
        continue; 
      /* only want 1k hour */ 
      tot += s_FIC[i].wnoduff;
      if ( !strcmp ( s_FIC[i].cr_FS,"F" ) )  
         continue;  /* don't want Flame */   
      smo += s_FIC[i].wnoduff;   /* total smolering */ 
}
    if ( smo == 0 ) 
      f = 0 ;
    else 
      f = smo / tot  ;   /* take percent */ 

    return f; 
} 

/********************************************************************
* Name: FIC_Put
* Desc: for the current time step save each compoents info
*       The emission function can then see if an individual
*       component went up in flame or smolder
* Note-1 wnoduff - wood no duff, not sure if this is the actual fuel
*         consumed or not, but might use to get a percent of flame/smolder 
* Note-1: was have trouble compare doubles, 
*********************************************************************/
void FIC_Put (double d_SURat, double d_dendry,  double term, double test, double wnoduff, char cr_FS[])
{
int i;
    if ( d_SURat < (e_SURat_DWk_3_6 + 1.0) )  /* See Note-1 - add 1.0 to help compare */
       goto Large;

    for ( i  = 0; i < 20; i++ ) {
      if ( _isEqual ( s_FIC[i].d_SIRat,d_SURat) ) {  /* See Note-1 */
        s_FIC[i].term = term;
        s_FIC[i].test = test;
        s_FIC[i].wnoduff = wnoduff; 
        strcpy (s_FIC[i].cr_FS , cr_FS);
        return; }
    }

i++; 


Large:
    for ( i  = 0; i < 20; i++ ) {
      if ( _isEqual (s_FIC[i].d_SIRat,d_SURat) &&  
           _isEqual (s_FIC[i].d_dendry,d_dendry)  ) {
         s_FIC[i].term = term;
         s_FIC[i].test = test;
         s_FIC[i].wnoduff = wnoduff;          /* Note-1 */
         strcpy (s_FIC[i].cr_FS , cr_FS);
         return;  }
    } 
  i++; 

}

/**************************************************************************
* Used to compare two doubles 
* I noticed when I was comparing equal 'sigma' values they didn't match
*  they were off by really small decimal amounts
* It might because I used a float for sigma(double) values when load then into
*  burnup, 
***************************************************************************/
bool _isEqual (double a, double b)
{
double d;

  d = abs (long(a - b));
 
  if ( d < 1.0 )   // was 0.2
    return true;
  return false ;
 
}

/******************************************************************
*
  e_SURat_Lit       8200.0
  e_SURat_DW1       1480.0
  e_SURat_DW10       394.0
  e_SURat_DW100      105.0
  e_SURat_DWk_3_6     39.4
  e_SURat_DWk_6_9     21.9
  e_SURat_DWk_9_20    12.7
  e_SURat_DWk_20      5.91
*
*******************************************************************/
typedef struct {
    float FLit,   SLit,   TLit,   MLit,   ALit ;  
    float FHr1,   SHr1,   THr1,   MHr1,   AHr1;
    float FHr10,  SHr10,  THr10,  MHr10,  AHr10; 
    float FHr100, SHr100, THr100, MHr100, AHr100; 


    float F3In,  S3In,   T3In,  M3In,  A3In; 
    float F6In,  S6In,   T6In,  M6In,  A6In; 
    float F9In,  S9In,   T9In,  M9In,  A9In; 
    float F20In,  S20In, T20In, M20In, A20In;    
  } d_CD;

d_CD s_CD; 

int i_CDTime = 0;  

FILE *fhCD; 

/***************************************************************************
* Name: _CompDump
* Desc: Test Code, to dump out FireIntensity() variables to a file
*       This one function is used to do all the various functions needed
*       included opening and closing file, see code below
* NOTE NOTE: Set percent rotten on GUI to 0 - so there will only be one 
*            fuel compent for each of the 1000 Hour 3,6,9,20 or else
*            there will be two for each and they will over write each
*            other when they get saved (see below)
* NOTE: the file with no path used will go into \FOF_GUI
*  In: see caller
*      wnoduff.... doesn't appear to be the fuel load but we save and printf it out
*                  anyway.
*************************************************************************/
void _CompDump (float test, float cmpark,  float term, char FS[], float wnoduff, double sigma)
{
int i = 0;
int j = 0;

// F-62 completion/acceptance-recovery pass (2026-09-21) DIAGNOSTIC HOOK.
// The pinned code's unconditional 'return' here (see the ORIGINAL
// comment immediately below, preserved verbatim) permanently disables
// this whole pre-existing per-timestep/per-fuel-class flame-vs-smolder
// dump facility. This is the ONLY line changed in this file relative to
// the pinned reference/fofem_cpp/FOF_UNIX/bur_brn.cpp (verified via a
// byte-identical copy made before this edit) -- gated behind an opt-in
// environment variable so normal (non-diagnostic) behavior, including
// every existing golden/self-test run, is completely unaffected: the
// env var is never set outside this one investigative use, and even
// when set, this function's own logic below (a pre-existing, already-
// authored-by-the-original-FOFEM-developers CSV dump) is otherwise
// untouched -- no equation, no arithmetic, and no classification/
// simulation state is changed anywhere in this file.
  if ( getenv("FOFEM_TEST_BURN_DIAG") == NULL )
    return;

// putting a 'return' will turn off all calls, so no file will get opened, etc.
// MUST have this 'return' here when building install, FS computer can't have
//  files going to program folder,
// (ORIGINAL unconditional 'return ;' preserved as a comment, not code,
//  immediately above -- this diagnostic build intentionally reaches
//  past it only when FOFEM_TEST_BURN_DIAG is set)

   if ( !strcasecmp (FS,"S"))
     goto Save;
   if ( !strcasecmp (FS,"F"))
     goto Save; 

/*.......................................................*/
/*   Open File and put headings in file  */ 
   if ( !strcasecmp (FS, "Open") ) {
      i_CDTime = 0; 
      fhCD = fopen ("_FireInt-Dump.csv", "w"); 
      if ( fhCD == NULL ) 
        return; 
      fprintf (fhCD, "\nNOTE, ->>, Rotten, MUST, be, set, to, 0, See, Notes, in, code, \n\n"); 
      fprintf (fhCD, "Seconds,FI-Total,"); 
      fprintf (fhCD, "Tot-Fla+Smo,Per-Smo,,");
      fprintf (fhCD, "Lit-F,Lit-S,Lit-tst,>=,Lit-trm,,"); 
      fprintf (fhCD, "1Hr-F,1Hr-S,1Hr-tst,>=,1Hr-trm,,"); 
      fprintf (fhCD, "10Hr-F,10Hr-S,10Hr-tst,>=,10Hr-trm,,"); 
      fprintf (fhCD, "100Hr-F,100Hr-S,100Hr-tst,>=,100Hr-trm,,");  
      fprintf (fhCD, "3hr-F,3hr-S,3Hr-tst,>=,3Hr-trm,,"); 
      fprintf (fhCD, "6hr-F,6hr-S,6Hr-tst,>=,6Hr-trm,,"); 
      fprintf (fhCD, "9hr-F,9hr-S,9Hr-tst,>=,9Hr-trm,,"); 
      fprintf (fhCD, "20hr-F,20hr-S,20Hr-tst,>=,20Hr-trm\n"); 
      return;  } 
    
/*...................................................*/
   if ( !strcasecmp (FS, "Close") ) {
     fclose (fhCD);
     return;   } 

/*..................................................*/
/* Init the struct for each time step    */
   if ( !strcasecmp (FS, "Init" ) ) {
     s_CD.FLit =   s_CD.SLit = s_CD.TLit = 0;
     s_CD.FHr1 =   s_CD.SHr1 = s_CD.THr1 = 0;
     s_CD.FHr10 =  s_CD.SHr10 = s_CD.THr10 = 0; 
     s_CD.FHr100 = s_CD.SHr100 = s_CD.THr100 = 0; 

     s_CD.F3In =  s_CD.S3In = s_CD.T3In = 0; 
     s_CD.F6In =  s_CD.S6In = s_CD.T6In = 0; 
     s_CD.F9In =  s_CD.S9In = s_CD.T9In = 0; 
     s_CD.F20In = s_CD.S20In = s_CD.T20In = 0; 
     return ;  }

/*...................................................*/
/* Put the information for one time step             */
   if ( !strcasecmp (FS, "Put" ) ) {
     if ( i_CDTime == 0 )   /* Keep track of time steps */  
        i_CDTime = 60;
     else 
        i_CDTime += 15; 

     fprintf (fhCD, "%d", i_CDTime); 
/* on "Put" these values are different sent in with differnt values than the param names - see caller */
     fprintf (fhCD, ",%f", test);      /* 'Sum' from FireIntensity() */
     fprintf (fhCD, ",%f", wnoduff);   /* total  Flame + Smolder*/
     fprintf (fhCD, ",%f,", sigma);     /* percent Smolder */

     fprintf (fhCD,",%6.8f,%6.8f,%6.4f,%6.4f,%6.4f,", s_CD.FLit,   s_CD.SLit,   s_CD.TLit,    s_CD.ALit,      s_CD.MLit     );  
                                                                                                            
     fprintf (fhCD,",%6.8f,%6.8f,%6.4f,%6.4f,%6.4f,", s_CD.FHr1,   s_CD.SHr1,   s_CD.THr1,    s_CD.AHr1,      s_CD.MHr1      );  
     fprintf (fhCD,",%6.8f,%6.8f,%6.4f,%6.4f,%6.4f,", s_CD.FHr10,  s_CD.SHr10,  s_CD.THr10,   s_CD.AHr10,     s_CD.MHr10      );  
     fprintf (fhCD,",%6.8f,%6.8f,%6.4f,%6.4f,%6.4f,", s_CD.FHr100, s_CD.SHr100, s_CD.THr100,  s_CD.AHr100,    s_CD.MHr100       );  
                                                                                                            
     fprintf (fhCD,",%6.8f,%6.8f,%6.4f,%6.4f,%6.4f,", s_CD.F3In,  s_CD.S3In,    s_CD.T3In,    s_CD.A3In,      s_CD.M3In      );  
     fprintf (fhCD,",%6.8f,%6.8f,%6.4f,%6.4f,%6.4f,", s_CD.F6In,  s_CD.S6In,    s_CD.T6In,    s_CD.A6In,      s_CD.M6In      ); 
     fprintf (fhCD,",%6.8f,%6.8f,%6.4f,%6.4f,%6.4f,", s_CD.F9In,  s_CD.S9In,    s_CD.T9In,    s_CD.A9In,      s_CD.M9In      ); 
     fprintf (fhCD,",%6.8f,%6.8f,%6.4f,%6.4f,%6.4f",  s_CD.F20In, s_CD.S20In,    s_CD.T20In,   s_CD.A20In,     s_CD.M20In       ); 
     fprintf (fhCD, "\n"); 
     return;   } 


/*------------------------------------------------------------------------*/
/* Store the data for each fuel component                                 */
/* use sigma to determine fuel component, + 0.0001 help comparision work correctly */
Save: 
double  d_SURat_Lit      = e_SURat_Lit     + 0.001;
double  d_SURat_DW1      = e_SURat_DW1     + 0.001;
double  d_SURat_DW10     = e_SURat_DW10    + 0.001;
double  d_SURat_DW100    = e_SURat_DW100   + 0.001;
double  d_SURat_DWk_3_6  = e_SURat_DWk_3_6 + 0.001;
double  d_SURat_DWk_6_9  = e_SURat_DWk_6_9   + 0.001; 
double  d_SURat_DWk_9_20 = e_SURat_DWk_9_20  + 0.001; 
double  d_SURat_DWk_20   = e_SURat_DWk_20    + 0.001; 

  if ( sigma <= d_SURat_DWk_20   ) {
     if ( !strcasecmp (FS, "F" )) 
        s_CD.F20In = wnoduff;    
     else 
        s_CD.S20In = wnoduff;
     s_CD.T20In = test;   
     s_CD.M20In = term; 
     s_CD.A20In = cmpark; }

  else if ( sigma <= d_SURat_DWk_9_20   ) {
     if ( !strcasecmp (FS, "F" )) 
        s_CD.F9In = wnoduff; 
     else 
        s_CD.S9In = wnoduff;
     s_CD.T9In = test;   
     s_CD.M9In = term; 
     s_CD.A9In = cmpark;  }

  else if ( sigma <= d_SURat_DWk_6_9   ) {
     if ( !strcasecmp (FS, "F" )) 
        s_CD.F6In = wnoduff; 
     else 
        s_CD.S6In = wnoduff;
     s_CD.T6In = test;   
     s_CD.M6In = term; 
     s_CD.A6In = cmpark; }

  else if ( sigma <= d_SURat_DWk_3_6   ) {
     if ( !strcasecmp (FS, "F" )) 
        s_CD.F3In = wnoduff; 
     else 
        s_CD.S3In = wnoduff;
     s_CD.T3In = test;   
     s_CD.M3In = term; 
     s_CD.A3In = cmpark;  }

/* 100 Hour */ 
  else if ( sigma <= d_SURat_DW100  ) {
     if ( !strcasecmp (FS, "F" )) 
        s_CD.FHr100 = wnoduff; 
     else 
        s_CD.SHr100 = wnoduff;
     s_CD.THr100 = test;   
     s_CD.MHr100 = term; 
     s_CD.AHr100 = cmpark;   }

/* 10 Hour */ 
  else if ( sigma <= d_SURat_DW10  ) {
     if ( !strcasecmp (FS, "F" )) 
        s_CD.FHr10 = wnoduff; 
     else 
        s_CD.SHr10 = wnoduff;
     s_CD.THr10 = test;   
     s_CD.MHr10 = term; 
     s_CD.AHr10 = cmpark; }

/* 1 Hour */ 
  else if ( sigma <= d_SURat_DW1  ) {
     if ( !strcasecmp (FS, "F" )) 
        s_CD.FHr1 = wnoduff; 
     else 
        s_CD.SHr1 = wnoduff;
     s_CD.THr1 = test;   
     s_CD.MHr1 = term;
     s_CD.AHr1 = cmpark; }

/* Litter */ 
  else if ( sigma <= d_SURat_Lit  ) {
     if ( !strcasecmp (FS, "F" )) 
        s_CD.FLit = wnoduff; 
     else 
        s_CD.SLit = wnoduff;
     s_CD.TLit = test;   
     s_CD.MLit = term;  
     s_CD.ALit = cmpark; }

  else
     i++; /* error */ 

#ifdef owoowo
   else if ( sigma <= d_SURat_DW100  ) {
     a_ES->d_rm_100Hr += d_Rem;
     a_ES->d_cn_100Hr += d_Con; }

   else if ( sigma <= d_SURat_DW10  ) {
     a_ES->d_rm_10Hr += d_Rem;
     a_ES->d_cn_10Hr += d_Con; }

   else if ( sigma <= d_SURat_DW1  ) {
     a_ES->d_rm_1Hr += d_Rem;
     a_ES->d_cn_1Hr += d_Con; }

   else {
     a_ES->d_rm_Lit += d_Rem;
     a_ES->d_cn_Lit += d_Con; }

#endif 



} 




#ifdef FOFEM
//  put this in for testing FOFEM, DL wanted a GUI combobox
//  to fool with Duff fire intenstiy
/************************************************************
* 
* In: Rm.... Ratio of moisture mass to dry organic mass
*            ex: Dry = 0.40, Wet = 1.30 
**************************************************************/
float  _FireInt_DuffType (double Rm)
{
float f;

  f = 11.25 - (4.05 * Rm) ;   // default, but should always get a match below 


  if ( !strcasecmp ("PeatLow", cr_DuffType ) )   //   �Peat Low: (15 MJ/kg)�: Int = 11.25  � (4.05*Rm)
    f = 11.25 - (4.05 * Rm); 

  if ( !strcasecmp ("PeatHigh", cr_DuffType))    //   Peat High (20 MJ/kg)�: Int = 15.00  � (5.40*Rm)   
    f = 15.00 - (5.40 * Rm); 

  if ( !strcasecmp ("DuffLow", cr_DuffType))     //   Duff Low (25 MJ/kg)�: Int = 18.75  � (6.75*Rm)  
    f = 18.75 - (6.75 * Rm); 

  if ( !strcasecmp ("DuffHigh", cr_DuffType))    //  Duff High  (30 MJ/kg)�: Int = 22.50  � (8.10*Rm)   
    f = 22.50 - ( 8.10 * Rm ); 

  return f; 
} 
#endif
