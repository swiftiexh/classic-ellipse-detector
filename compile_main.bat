call "D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d E:\scc_desk\Net\classic-ellipse-detector
cl /std:c++17 /EHsc /MD /nologo /I "E:\OPENCV\opencv\build\include" /I "src" /D "AAMED_OPENCV_PROJECT_ROOT=E:\\scc_desk\\Net\\classic-ellipse-detector" /c /Fo:"build/main.obj" "src/main.cpp"
