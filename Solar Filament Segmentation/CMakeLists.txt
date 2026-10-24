# CMakeLists.txt del Subproyecto (Dentro de la subcarpeta)

set(FUENTES "Solar Filament Segmentation.cpp")

# Creamos el ejecutable
add_executable(mi_programa_torc ${FUENTES} maskApi.c)

# Forzamos a que este target conozca las cabeceras de ambos mundos explícitamente
target_include_directories(mi_programa_torc PRIVATE ${TORCH_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# Enlazamos las librerías binarias (.lib)
target_link_libraries(mi_programa_torc PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
