@echo off
call "D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to setup MSVC environment
    exit /b 1
)

set ROOT=E:\scc_desk\Net\classic-ellipse-detector
set OPENCV=E:\OPENCV\opencv\build
set INCLUDE=%OPENCV%\include;%ROOT%\src;%INCLUDE%
set LIB=%OPENCV%\x64\vc16\lib;%LIB%

set CFLAGS=/std:c++17 /O2 /EHsc /MD /nologo /D AAMED_OPENCV_PROJECT_ROOT="\"E:\\scc_desk\\Net\\classic-ellipse-detector\""
set LDFLAGS=opencv_world4120.lib

mkdir "%ROOT%\build\bin" 2>nul

echo Compiling...
cd /d "%ROOT%"

cl %CFLAGS% /c /Fo:"build\adaptApproxPolyDP.obj" "src\adaptApproxPolyDP.cpp"
if errorlevel 1 exit /b 1

cl %CFLAGS% /c /Fo:"build\Contours.obj" "src\Contours.cpp"
if errorlevel 1 exit /b 1

cl %CFLAGS% /c /Fo:"build\EllipseNonMaximumSuppression.obj" "src\EllipseNonMaximumSuppression.cpp"
if errorlevel 1 exit /b 1

cl %CFLAGS% /c /Fo:"build\FLED.obj" "src\FLED.cpp"
if errorlevel 1 exit /b 1

cl %CFLAGS% /c /Fo:"build\FLED_Export.obj" "src\FLED_Export.cpp"
if errorlevel 1 exit /b 1

cl %CFLAGS% /c /Fo:"build\FLED_drawAndWriteFunctions.obj" "src\FLED_drawAndWriteFunctions.cpp"
if errorlevel 1 exit /b 1

cl %CFLAGS% /c /Fo:"build\FLED_Initialization.obj" "src\FLED_Initialization.cpp"
if errorlevel 1 exit /b 1

cl %CFLAGS% /c /Fo:"build\FLED_PrivateFunctions.obj" "src\FLED_PrivateFunctions.cpp"
if errorlevel 1 exit /b 1

cl %CFLAGS% /c /Fo:"build\Group.obj" "src\Group.cpp"
if errorlevel 1 exit /b 1

cl %CFLAGS% /c /Fo:"build\LinkMatrix.obj" "src\LinkMatrix.cpp"
if errorlevel 1 exit /b 1

cl %CFLAGS% /c /Fo:"build\Node_FC.obj" "src\Node_FC.cpp"
if errorlevel 1 exit /b 1

cl %CFLAGS% /c /Fo:"build\Segmentation.obj" "src\Segmentation.cpp"
if errorlevel 1 exit /b 1

cl %CFLAGS% /c /Fo:"build\Validation.obj" "src\Validation.cpp"
if errorlevel 1 exit /b 1

cl %CFLAGS% /c /Fo:"build\main.obj" "src\main.cpp"
if errorlevel 1 exit /b 1

echo Linking aamed_demo.exe...
link /nologo /OUT:"build\bin\aamed_demo.exe" ^
    build\adaptApproxPolyDP.obj ^
    build\Contours.obj ^
    build\EllipseNonMaximumSuppression.obj ^
    build\FLED.obj ^
    build\FLED_Export.obj ^
    build\FLED_drawAndWriteFunctions.obj ^
    build\FLED_Initialization.obj ^
    build\FLED_PrivateFunctions.obj ^
    build\Group.obj ^
    build\LinkMatrix.obj ^
    build\Node_FC.obj ^
    build\Segmentation.obj ^
    build\Validation.obj ^
    build\main.obj ^
    %LDFLAGS%
if errorlevel 1 exit /b 1

echo Build SUCCESS: build\bin\aamed_demo.exe
