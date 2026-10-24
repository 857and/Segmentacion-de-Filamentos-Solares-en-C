# CMakeLists.txt de nivel superior (Raíz)
cmake_minimum_required (VERSION 3.18)

project ("Solar Filament Segmentation" C CXX CUDA)

# Configuración global de C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Buscar LibTorch
set(CMAKE_PREFIX_PATH "D:/Programas/libTorch/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Buscar OpenCV 5
set(OpenCV_DIR "D:/Programas/openCV/opencv/build")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Incluir el subproyecto (Esto activa la subcarpeta automáticamente)
add_subdirectory ("Solar Filament Segmentation")