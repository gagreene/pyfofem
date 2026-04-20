@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
cl /nologo /EHsc /O2 /std:c++14 /IFOF_UNIX FOF_UNIX\test_harness.cpp FOF_UNIX\fof_cm.cpp FOF_UNIX\fof_ci.cpp FOF_UNIX\fof_co.cpp FOF_UNIX\fof_bcm.cpp FOF_UNIX\bur_brn.cpp FOF_UNIX\bur_bov.cpp FOF_UNIX\fof_hsf.cpp FOF_UNIX\fof_duf.cpp FOF_UNIX\fof_nes.cpp FOF_UNIX\fof_util.cpp FOF_UNIX\fof_lem.cpp FOF_UNIX\fof_sgv.cpp FOF_UNIX\fof_disp.cpp FOF_UNIX\fof_unix.cpp FOF_UNIX\fof_sh.cpp FOF_UNIX\fof_sha.cpp FOF_UNIX\fof_soi.cpp FOF_UNIX\fof_sd.cpp FOF_UNIX\fof_se.cpp FOF_UNIX\fof_mrt.cpp FOF_UNIX\cdf_util.cpp /Fe:fofem_test.exe > compile_log.txt 2>&1
echo EXITCODE=%ERRORLEVEL% >> compile_log.txt
exit /b %ERRORLEVEL%
