/*
 * test_harness.cpp - Parameterized FOFEM test harness.
 * Reads a CSV input file, runs CM_Mngr for each row, writes structured CSV output.
 *
 * Usage: fofem_test <input.csv> <output_prefix>
 * Produces: <output_prefix>_components.csv and <output_prefix>_summary.csv
 *
 * Input CSV columns (header row required):
 *   litter,duff,duff_depth,duff_moist,herb,shrub,
 *   crown_fol,crown_bra,pct_crown_burn,
 *   dw10_moist,dw1000_moist,
 *   dw1,dw10,dw100,
 *   snd_dw3,snd_dw6,snd_dw9,snd_dw20,
 *   rot_dw3,rot_dw6,rot_dw9,rot_dw20,
 *   region,season,fuel_cat,
 *   intensity,ig_time,windspeed,depth,ambient_temp,
 *   cri_int
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fof_sgv.h"
#include "fof_ci.h"
#include "fof_co.h"
#include "fof_co2.h"
#include "fof_cm.h"
#include "fof_ansi.h"
#include "fof_sh.h"
#include "fof_nes.h"
#include "fof_duf.h"
#include "fof_hsf.h"

/* Max line length for CSV parsing */
#define MAX_LINE 4096
#define MAX_FIELD 256

/* Simple CSV field parser - reads next comma-delimited field */
static char *next_field(char **cursor) {
    static char buf[MAX_FIELD];
    char *start = *cursor;
    if (!start || *start == '\0') return NULL;
    char *end = strchr(start, ',');
    if (end) {
        int len = (int)(end - start);
        if (len >= MAX_FIELD) len = MAX_FIELD - 1;
        strncpy(buf, start, len);
        buf[len] = '\0';
        *cursor = end + 1;
    } else {
        /* Last field - trim newline */
        strncpy(buf, start, MAX_FIELD - 1);
        buf[MAX_FIELD - 1] = '\0';
        char *nl = strchr(buf, '\n');
        if (nl) *nl = '\0';
        nl = strchr(buf, '\r');
        if (nl) *nl = '\0';
        *cursor = NULL;
    }
    return buf;
}

static float field_float(char **cursor) {
    char *f = next_field(cursor);
    return f ? (float)atof(f) : 0.0f;
}

static void field_str(char **cursor, char *dest, int maxlen) {
    char *f = next_field(cursor);
    if (f) {
        strncpy(dest, f, maxlen - 1);
        dest[maxlen - 1] = '\0';
    } else {
        dest[0] = '\0';
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        printf("Usage: fofem_test <input.csv> <output_prefix>\n");
        return 1;
    }

    char *in_path = argv[1];
    char *out_prefix = argv[2];

    FILE *fin = fopen(in_path, "r");
    if (!fin) { printf("Cannot open %s\n", in_path); return 1; }

    /* Build output file paths */
    char comp_path[512], summ_path[512];
    sprintf(comp_path, "%s_components.csv", out_prefix);
    sprintf(summ_path, "%s_summary.csv", out_prefix);

    FILE *fcomp = fopen(comp_path, "w");
    FILE *fsumm = fopen(summ_path, "w");
    if (!fcomp || !fsumm) { printf("Cannot open output files\n"); return 1; }

    /* Component output header */
    fprintf(fcomp, "case,component,pre_tac,con_tac,pos_tac,pct_con,equation\n");

    /* Summary output header */
    fprintf(fsumm, "case,"
        "LitPre,LitCon,LitPos,"
        "DW1Pre,DW1Con,DW1Pos,"
        "DW10Pre,DW10Con,DW10Pos,"
        "DW100Pre,DW100Con,DW100Pos,"
        "SndDW1kPre,SndDW1kCon,SndDW1kPos,"
        "RotDW1kPre,RotDW1kCon,RotDW1kPos,"
        "DufPre,DufCon,DufPos,DufPer,"
        "HerPre,HerCon,HerPos,"
        "ShrPre,ShrCon,ShrPos,"
        "FolPre,FolCon,FolPos,"
        "BraPre,BraCon,BraPos,"
        "TotPre,TotCon,TotPos,"
        "FlaCon,SmoCon,FlaDur,SmoDur,"
        "PM25F,PM25S,PM10F,PM10S,CH4F,CH4S,COF,COS,CO2F,CO2S,NOXF,NOXS,SO2F,SO2S,"
        "MSE,DufDepPre,DufDepCon,DufDepPos,"
        "ret_code\n");

    /* Skip header line */
    char line[MAX_LINE];
    if (!fgets(line, MAX_LINE, fin)) { printf("Empty input\n"); return 1; }

    int case_num = 0;
    while (fgets(line, MAX_LINE, fin)) {
        case_num++;
        char *cursor = line;

        d_CI s_CI;
        d_CO s_CO;
        char cr_ErrMes[3000];

        CI_Init(&s_CI);
        CO_Init(&s_CO);

        /* Parse CSV fields */
        s_CI.f_Lit           = field_float(&cursor);
        s_CI.f_Duff          = field_float(&cursor);
        s_CI.f_DufDep        = field_float(&cursor);
        s_CI.f_MoistDuff     = field_float(&cursor);
        s_CI.f_Herb          = field_float(&cursor);
        s_CI.f_Shrub         = field_float(&cursor);
        s_CI.f_CroFol        = field_float(&cursor);
        s_CI.f_CroBra        = field_float(&cursor);
        s_CI.f_Pc_CroBrn     = field_float(&cursor);
        s_CI.f_MoistDW10     = field_float(&cursor);
        s_CI.f_MoistDW1000   = field_float(&cursor);
        s_CI.f_DW1           = field_float(&cursor);
        s_CI.f_DW10          = field_float(&cursor);
        s_CI.f_DW100         = field_float(&cursor);
        s_CI.f_Snd_DW3       = field_float(&cursor);
        s_CI.f_Snd_DW6       = field_float(&cursor);
        s_CI.f_Snd_DW9       = field_float(&cursor);
        s_CI.f_Snd_DW20      = field_float(&cursor);
        s_CI.f_Rot_DW3       = field_float(&cursor);
        s_CI.f_Rot_DW6       = field_float(&cursor);
        s_CI.f_Rot_DW9       = field_float(&cursor);
        s_CI.f_Rot_DW20      = field_float(&cursor);

        char region[50], season[50], fuel_cat[50];
        field_str(&cursor, region, 50);
        field_str(&cursor, season, 50);
        field_str(&cursor, fuel_cat, 50);

        strcpy(s_CI.cr_Region, region);
        strcpy(s_CI.cr_Season, season);
        strcpy(s_CI.cr_FuelCategory, fuel_cat);
        strcpy(s_CI.cr_DufMoiMet, ENTIRE);

        /* Fire environment overrides (use CI_Init defaults if <= 0) */
        float v;
        v = field_float(&cursor); if (v > 0) s_CI.f_INTENSITY = v;
        v = field_float(&cursor); if (v > 0) s_CI.f_IG_TIME = v;
        v = field_float(&cursor); if (v >= 0) s_CI.f_WINDSPEED = v;
        v = field_float(&cursor); if (v > 0) s_CI.f_DEPTH = v;
        v = field_float(&cursor); if (v != 0) s_CI.f_AMBIENT_TEMP = v;

        /* Emission mode: -1 = original, 15 = expanded */
        float cri = field_float(&cursor);
        s_CI.f_CriInt = cri;

        /* Don't write output files per case */
        strcpy(s_CI.cr_LoadFN, "");
        strcpy(s_CI.cr_EmiFN, "");

        /* Run */
        int ret = CM_Mngr(&s_CI, &s_CO, cr_ErrMes);

        /* Write component outputs */
        fprintf(fcomp, "%d,Litter,%.6f,%.6f,%.6f,%.4f,%d\n", case_num,
            s_CO.f_LitPre, s_CO.f_LitCon, s_CO.f_LitPos, s_CO.f_LitPer, s_CO.i_LitEqu);
        fprintf(fcomp, "%d,DW1,%.6f,%.6f,%.6f,%.4f,%d\n", case_num,
            s_CO.f_DW1Pre, s_CO.f_DW1Con, s_CO.f_DW1Pos, s_CO.f_DW1Per, s_CO.i_DW1Equ);
        fprintf(fcomp, "%d,DW10,%.6f,%.6f,%.6f,%.4f,%d\n", case_num,
            s_CO.f_DW10Pre, s_CO.f_DW10Con, s_CO.f_DW10Pos, s_CO.f_DW10Per, s_CO.i_DW10Equ);
        fprintf(fcomp, "%d,DW100,%.6f,%.6f,%.6f,%.4f,%d\n", case_num,
            s_CO.f_DW100Pre, s_CO.f_DW100Con, s_CO.f_DW100Pos, s_CO.f_DW100Per, s_CO.i_DW100Equ);
        fprintf(fcomp, "%d,SndDW1k,%.6f,%.6f,%.6f,%.4f,%d\n", case_num,
            s_CO.f_Snd_DW1kPre, s_CO.f_Snd_DW1kCon, s_CO.f_Snd_DW1kPos, s_CO.f_Snd_DW1kPer, s_CO.i_Snd_DW1kEqu);
        fprintf(fcomp, "%d,RotDW1k,%.6f,%.6f,%.6f,%.4f,%d\n", case_num,
            s_CO.f_Rot_DW1kPre, s_CO.f_Rot_DW1kCon, s_CO.f_Rot_DW1kPos, s_CO.f_Rot_DW1kPer, s_CO.i_Rot_DW1kEqu);
        fprintf(fcomp, "%d,Duff,%.6f,%.6f,%.6f,%.4f,%d\n", case_num,
            s_CO.f_DufPre, s_CO.f_DufCon, s_CO.f_DufPos, s_CO.f_DufPer, s_CO.i_DufEqu);
        fprintf(fcomp, "%d,Herb,%.6f,%.6f,%.6f,%.4f,%d\n", case_num,
            s_CO.f_HerPre, s_CO.f_HerCon, s_CO.f_HerPos, s_CO.f_HerPer, s_CO.i_HerEqu);
        fprintf(fcomp, "%d,Shrub,%.6f,%.6f,%.6f,%.4f,%d\n", case_num,
            s_CO.f_ShrPre, s_CO.f_ShrCon, s_CO.f_ShrPos, s_CO.f_ShrPer, s_CO.i_ShrEqu);
        fprintf(fcomp, "%d,Foliage,%.6f,%.6f,%.6f,%.4f,%d\n", case_num,
            s_CO.f_FolPre, s_CO.f_FolCon, s_CO.f_FolPos, s_CO.f_FolPer, s_CO.i_FolEqu);
        fprintf(fcomp, "%d,Branch,%.6f,%.6f,%.6f,%.4f,%d\n", case_num,
            s_CO.f_BraPre, s_CO.f_BraCon, s_CO.f_BraPos, s_CO.f_BraPer, s_CO.i_BraEqu);

        /* Write summary row */
        fprintf(fsumm, "%d,"
            "%.6f,%.6f,%.6f,"   /* Lit */
            "%.6f,%.6f,%.6f,"   /* DW1 */
            "%.6f,%.6f,%.6f,"   /* DW10 */
            "%.6f,%.6f,%.6f,"   /* DW100 */
            "%.6f,%.6f,%.6f,"   /* SndDW1k */
            "%.6f,%.6f,%.6f,"   /* RotDW1k */
            "%.6f,%.6f,%.6f,%.4f,"  /* Duf + pct */
            "%.6f,%.6f,%.6f,"   /* Her */
            "%.6f,%.6f,%.6f,"   /* Shr */
            "%.6f,%.6f,%.6f,"   /* Fol */
            "%.6f,%.6f,%.6f,"   /* Bra */
            "%.6f,%.6f,%.6f,"   /* Tot */
            "%.6f,%.6f,%.1f,%.1f,"  /* FlaCon SmoCon FlaDur SmoDur */
            "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"  /* emissions */
            "%.4f,%.4f,%.4f,%.4f,"  /* MSE DufDep */
            "%d\n",
            case_num,
            s_CO.f_LitPre, s_CO.f_LitCon, s_CO.f_LitPos,
            s_CO.f_DW1Pre, s_CO.f_DW1Con, s_CO.f_DW1Pos,
            s_CO.f_DW10Pre, s_CO.f_DW10Con, s_CO.f_DW10Pos,
            s_CO.f_DW100Pre, s_CO.f_DW100Con, s_CO.f_DW100Pos,
            s_CO.f_Snd_DW1kPre, s_CO.f_Snd_DW1kCon, s_CO.f_Snd_DW1kPos,
            s_CO.f_Rot_DW1kPre, s_CO.f_Rot_DW1kCon, s_CO.f_Rot_DW1kPos,
            s_CO.f_DufPre, s_CO.f_DufCon, s_CO.f_DufPos, s_CO.f_DufPer,
            s_CO.f_HerPre, s_CO.f_HerCon, s_CO.f_HerPos,
            s_CO.f_ShrPre, s_CO.f_ShrCon, s_CO.f_ShrPos,
            s_CO.f_FolPre, s_CO.f_FolCon, s_CO.f_FolPos,
            s_CO.f_BraPre, s_CO.f_BraCon, s_CO.f_BraPos,
            s_CO.f_TotPre, s_CO.f_TotCon, s_CO.f_TotPos,
            s_CO.f_FlaCon, s_CO.f_SmoCon, s_CO.f_FlaDur, s_CO.f_SmoDur,
            s_CO.f_PM25F, s_CO.f_PM25S, s_CO.f_PM10F, s_CO.f_PM10S,
            s_CO.f_CH4F, s_CO.f_CH4S, s_CO.f_COF, s_CO.f_COS,
            s_CO.f_CO2F, s_CO.f_CO2S, s_CO.f_NOXF, s_CO.f_NOXS,
            s_CO.f_SO2F, s_CO.f_SO2S,
            s_CO.f_MSEPer, s_CO.f_DufDepPre, s_CO.f_DufDepCon, s_CO.f_DufDepPos,
            ret);

        if (ret == 0)
            printf("Case %d: ERROR - %s\n", case_num, cr_ErrMes);
        else if (ret == 2)
            printf("Case %d: No ignition\n", case_num);
        else
            printf("Case %d: OK (TotCon=%.4f T/ac)\n", case_num, s_CO.f_TotCon);
    }

    fclose(fin);
    fclose(fcomp);
    fclose(fsumm);
    printf("\nWrote %s and %s (%d cases)\n", comp_path, summ_path, case_num);
    return 0;
}
