

#include <iostream>
#include<torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem> 
namespace fs = std::filesystem;
#include <iomanip>


extern "C" {
#include "maskApi.h" 
}

#ifdef _WIN32
    // Rutas para tu PC Windows
    std::string base_path = "C:/Users/857an/Downloads/filament-segmentation-2026";
#else
    // Rutas para Linux (Colab)
    std::string base_path = "/content/filament-segmentation-2026";
#endif

constexpr int TARGET_SIZE = 512; //1024

struct DoubleConvImpl : torch::nn::Module {
	torch::nn::Sequential block;

	DoubleConvImpl(int64_t in_channels, int64_t out_channels) {
		block = register_module("block", torch::nn::Sequential(
			// Primera Convolución (Mantiene el tamaño de la imagen usando padding=1)
			torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1).bias(false)),
			torch::nn::BatchNorm2d(out_channels),
			torch::nn::Functional(torch::relu),

			// Segunda Convolución
			torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1).bias(false)),
			torch::nn::BatchNorm2d(out_channels),
			torch::nn::Functional(torch::relu)
		));
	}

	torch::Tensor forward(torch::Tensor x) {
		return block->forward(x);
	}
};
TORCH_MODULE(DoubleConv);


struct UNetImpl : torch::nn::Module {
    // Encoder (Camino Izquierdo)
    DoubleConv enc1{ nullptr }, enc2{ nullptr }, enc3{ nullptr }, enc4{ nullptr };
    // cuello de botella (Base de la U)
    DoubleConv bottleneck{ nullptr };

    // Decoder (Camino Derecho) y Convoluciones Transpuestas (Upsampling)
    torch::nn::ConvTranspose2d up1{ nullptr }, up2{ nullptr }, up3{ nullptr }, up4{ nullptr };
    DoubleConv dec1{ nullptr }, dec2{ nullptr }, dec3{ nullptr }, dec4{ nullptr };

    // Capa de Salida Final
    torch::nn::Conv2d final_conv{ nullptr };

    UNetImpl(int64_t in_channels = 1, int64_t out_channels = 1) { // 1 canal para imágenes H-Alpha del Sol en blanco y negro
        // Inicializar Encoder
        enc1 = register_module("enc1", DoubleConv(in_channels, 32));
        enc2 = register_module("enc2", DoubleConv(32, 64));
        enc3 = register_module("enc3", DoubleConv(64, 128));
        enc4 = register_module("enc4", DoubleConv(128, 256));

        // Inicializar Base
        bottleneck = register_module("bottleneck", DoubleConv(256, 512));

        // Inicializar Decoder (los canales se duplican cada vez aqui)
        up1 = register_module("up1", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(512, 256, 2).stride(2)));
        dec1 = register_module("dec1", DoubleConv(512, 256)); // 256 (up) + 256 (skip) = 512

        up2 = register_module("up2", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(256, 128, 2).stride(2)));
        dec2 = register_module("dec2", DoubleConv(256, 128));

        up3 = register_module("up3", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 2).stride(2)));
        dec3 = register_module("dec3", DoubleConv(128, 64));

        up4 = register_module("up4", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 32, 2).stride(2)));
        dec4 = register_module("dec4", DoubleConv(64, 32));

        // Convolución final 1x1 para reducir los 64 canales a la máscara binaria final (1 canal)
        final_conv = register_module("final_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, out_channels, 1)));
    }

    torch::Tensor forward(torch::Tensor x) {
        // --- ENCODER ---
        auto x1 = enc1->forward(x);
        auto p1 = torch::max_pool2d(x1, 2); // Reduce tamaño a la mitad

        auto x2 = enc2->forward(p1);
        auto p2 = torch::max_pool2d(x2, 2);

        auto x3 = enc3->forward(p2);
        auto p3 = torch::max_pool2d(x3, 2);

        auto x4 = enc4->forward(p3);
        auto p4 = torch::max_pool2d(x4, 2);


        // --- CUELLO DE BOTELLA ---
        auto b = bottleneck->forward(p4);

        // --- CAMINO DE EXPANSIÓN (DECODER + SKIP CONNECTIONS) ---
        // Nivel 1
        auto u1 = up1->forward(b);
        auto cat1 = torch::cat({ u1, x4 }, /*dim=*/1); // concatenar skip
        auto d1 = dec1->forward(cat1);

        // Nivel 2
        auto u2 = up2->forward(d1);
        auto cat2 = torch::cat({ u2, x3 }, /*dim=*/1);
        auto d2 = dec2->forward(cat2);

        // Nivel 3
        auto u3 = up3->forward(d2);
        auto cat3 = torch::cat({ u3, x2 }, /*dim=*/1);
        auto d3 = dec3->forward(cat3);

        auto u4 = up4->forward(d3);
        auto cat4 = torch::cat({ u4, x1 }, /*dim=*/1);
        auto d4 = dec4->forward(cat4);

        // --- CAPA DE SALIDA ---
        return final_conv->forward(d4); // Devuelve los logits de la máscara
    }
};
TORCH_MODULE(UNet);


#include <fstream>
#include <map>
#include "json.hpp" 

using json = nlohmann::json;

// --- ESTRUCTURA DEL DATASET PARA FORMATO COCO JSON ---
class SolarFilamentDataset : public torch::data::Dataset<SolarFilamentDataset> {
private:
    std::string images_dir;
    std::vector<std::string> image_filenames;
    // Mapea el nombre de la imagen a una lista de polígonos (cada polígono es un vector de puntos cv::Point)
    std::map<std::string, std::vector<std::vector<cv::Point>>> image_to_polygons;
    
    std::vector<torch::Tensor> cached_images;
    std::vector<torch::Tensor> cached_masks;

public:
    // El constructor ahora lee el JSON de Kaggle directamente
    SolarFilamentDataset(const std::string& json_path, const std::string& imgs_dir) : images_dir(imgs_dir) {
        std::ifstream file(json_path);
        if (!file.is_open()) {
            std::cerr << "❌ No se pudo abrir el archivo JSON en: " << json_path << std::endl;
            return;
        }

        json coco_data;
        file >> coco_data;

        // 1. Mapear image_id -> file_name
        std::map<std::string, std::string> id_to_filename;
        for (const auto& img : coco_data["images"]) {
            std::string img_id = img["id"].get<std::string>();
            std::string f_name = img["file_name"].get<std::string>();
            id_to_filename[img_id] = f_name;

            // Registrar los nombres únicos de las imágenes
            if (std::find(image_filenames.begin(), image_filenames.end(), f_name) == image_filenames.end()) {
                image_filenames.push_back(f_name);
            }
        }

        // 2. Leer las anotaciones y agrupar los polígonos por imagen
        for (const auto& ann : coco_data["annotations"]) {
            std::string img_id = ann["image_id"].get<std::string>();
            std::string f_name = id_to_filename[img_id];

            // COCO guarda los segmentos como una lista de listas de flotantes [x1, y1, x2, y2...]
            for (const auto& seg : ann["segmentation"]) {
                std::vector<float> coords = seg.get<std::vector<float>>();
                std::vector<cv::Point> polygon;

                for (size_t i = 0; i < coords.size(); i += 2) {
                    polygon.push_back(cv::Point(static_cast<int>(coords[i]), static_cast<int>(coords[i + 1])));
                }

                image_to_polygons[f_name].push_back(polygon);
            }
        }
        std::cout << "✅ Dataset cargado: " << image_filenames.size() << " imágenes encontradas en el JSON." << std::endl;
        std::cout << "⏳ Pre-procesando y cargando dataset en RAM..." << std::endl;

        for (const auto& filename : image_filenames) {

            std::string full_img_path = imgs_dir + "/" + filename;
            cv::Mat img = cv::imread(full_img_path, cv::IMREAD_GRAYSCALE);
            if (img.empty()) img = cv::Mat::zeros(2048, 2048, CV_8UC1);

            cv::Mat msk = cv::Mat::zeros(img.size(), CV_8UC1);
            if (image_to_polygons.find(filename) != image_to_polygons.end()) {
                cv::fillPoly(msk, image_to_polygons[filename], cv::Scalar(255));
            }

            // (x, y, ancho, alto) -> Centrado: (2048 - 1810) / 2 = 119
            cv::Rect roi(119, 119, 1810, 1810);

            img = img(roi).clone();
            msk = msk(roi).clone();

            //MÁSCARA CIRCULAR NEGRA: Creas una máscara blanca (255) del tamaño del recorte

            cv::Mat circular_mask = cv::Mat::zeros(img.size(), CV_8UC1);
            cv::Point center(img.cols / 2, img.rows / 2); // Centro: (905, 905)
            int radius = img.cols / 2;                    // Radio: 905

            // Dibujamos un círculo blanco relleno en la máscara
            cv::circle(circular_mask, center, radius, cv::Scalar(255), -1);

            // Aplicamos la máscara: lo que esté fuera del círculo se vuelve negro (0)
            cv::bitwise_and(img, circular_mask, img);
            cv::bitwise_and(msk, circular_mask, msk); // También a la máscara de filamentos por seguridad

            // Convertir a Tensor y guardar en el vector (caché)
            torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 1 }, torch::kByte).clone();
            torch::Tensor msk_tensor = torch::from_blob(msk.data, { msk.rows, msk.cols, 1 }, torch::kByte).clone();

            img_tensor = img_tensor.permute({ 2, 0, 1 });
            msk_tensor = msk_tensor.permute({ 2, 0, 1 }).gt(127).to(torch::kByte);

            cached_images.push_back(img_tensor);
            cached_masks.push_back(msk_tensor);
        }
        std::cout << "✅ Dataset cargado en RAM: " << cached_images.size() << " imágenes." << std::endl;
    
    }

    torch::optional<size_t> size() const override {
        return image_filenames.size();
    }

    torch::data::Example<> get(size_t index) override {
        // 1. Calcular origen (x,y) aleatorio para un recorte de TARGET_SIZE (512x512)
        int max_coord = 1810 - TARGET_SIZE;
        int x = torch::randint(0, max_coord + 1, { 1 }).item<int>();
        int y = torch::randint(0, max_coord + 1, { 1 }).item<int>();

        // 2. Extraer el recorte usando slicing de tensores
        using torch::indexing::Slice;
        torch::Tensor img_crop = cached_images[index].index({ 0, Slice(y, y + TARGET_SIZE), Slice(x, x + TARGET_SIZE) });
        torch::Tensor msk_crop = cached_masks[index].index({ 0, Slice(y, y + TARGET_SIZE), Slice(x, x + TARGET_SIZE) });

        // 3. Preparar tensores para la red
        // Como hicimos slice en dims {0, alto, ancho}, perdimos la dimensión de canales, así que hacemos unsqueeze(0)
        auto img_float = img_crop.unsqueeze(0).to(torch::kFloat32).div(255.0);

        // Como la máscara ya está en 0 y 1 (por el gt), NO la dividimos entre 255.0, solo la pasamos a Float32
        auto msk_float = msk_crop.unsqueeze(0).to(torch::kFloat32);

        return { img_float, msk_float };
    }
};

std::string maskToCOCORLE(const cv::Mat& mask) {
    // 1. COCO RLE lee en formato Column-Major (Fortran). Hay que trasponer la máscara.
    cv::Mat mask_t;
    cv::transpose(mask, mask_t);

    // 2. Asegurarnos de que sea continua y mapear los valores de 255 a 1 (binario estricto)
    cv::Mat mask_bin;
    mask_t.convertTo(mask_bin, CV_8UC1, 1.0 / 255.0);
    if (!mask_bin.isContinuous()) {
        mask_bin = mask_bin.clone();
    }
    // 3. Inicializar y codificar con tus archivos maskApi
    RLE rle;
    rleEncode(
        &rle,
        mask_bin.data,
        static_cast<siz>(mask.rows),
        static_cast<siz>(mask.cols),
        1
    );
    // convertir a string
    char* rle_str = rleToString(&rle);
    std::string resultado(rle_str);
    free(rle_str);
    rleFree(&rle);

    return resultado;
}

torch::Tensor dice_loss(torch::Tensor logits, torch::Tensor targets, float smooth = 1.0) {
    torch::Tensor probs = torch::sigmoid(logits);

    // Aplanar los tensores
    auto probs_flat = probs.reshape({-1});
    auto targets_flat = targets.reshape({-1});

    auto intersection = (probs_flat * targets_flat).sum();
    auto dice_score = (2. * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth);

    return 1.0 - dice_score;
}

torch::Tensor focal_loss(torch::Tensor logits, torch::Tensor targets, float alpha = 0.25, float gamma = 2.0) {
    torch::Tensor probs = torch::sigmoid(logits);
    // El cuarto argumento es la reducción. Pasamos tensores vacíos para weight y pos_weight.
    torch::Tensor ce_loss = torch::binary_cross_entropy_with_logits(
        logits,
        targets,
        /*weight=*/torch::Tensor(),
        /*pos_weight=*/torch::Tensor(),
        /*reduction=*/torch::Reduction::None
    );
    torch::Tensor p_t = torch::where(targets == 1.0, probs, 1.0 - probs);
    torch::Tensor focal_weight = torch::pow(1.0 - p_t, gamma);
    torch::Tensor focal = alpha * focal_weight * ce_loss;
    
    return focal.mean();
}

int main()
{	
    #ifdef _WIN32
    // En Windows forzamos el uso exclusivo de la CPU
        torch::Device device(torch::kCPU);
    #else
    // En Colab usamos la GPU si está disponible
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    #endif

    std::cout << "🖥️  Dispositivo en uso: " << device << std::endl;
    std::cout << "🚀 cuDNN disponible: " << (torch::cuda::cudnn_is_available() ? "Sí" : "No") << std::endl;

    if (torch::cuda::is_available()) {
        at::globalContext().setBenchmarkCuDNN(true);
    }

    std::string ruta_json = base_path + "/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json";
    std::string carpeta_imagenes = base_path + "/MAGFiLO_1.0_Kaggle_2026/train/train_images";

    auto solar_dataset = SolarFilamentDataset(ruta_json, carpeta_imagenes).map(torch::data::transforms::Stack<>());

    auto model = std::make_shared<UNetImpl>(1, 1); // 1 canal de entrada (grises), 1 canal de salida (máscara binaria)
    model->to(device);

    torch::optim::Adam optimizador(model->parameters(), 0.0005);
    auto criterion = torch::nn::BCEWithLogitsLoss();

	const int ciclos = 15;
	const int batch = 4;
    const int pasos_acumulacion = 2;

    size_t total_imagenes = solar_dataset.size().value();
    size_t total_lotes = (total_imagenes + batch - 1) / batch;

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(solar_dataset),
        torch::data::DataLoaderOptions().batch_size(batch).workers(0)
    );

    std::cout << "Empezando elentrenamiento..." << std::endl;


    for (int i = 1; i <= ciclos; i++) {
        float lossAcumulada = 0;
        int vueltas = 0; // Contador de lotes procesados

        model->train();
        model->zero_grad();
        for (auto& batch_data : *data_loader) {

            auto x_train = batch_data.data.to(device, /*non_blocking=*/true);
            auto y_train = batch_data.target.to(device, /*non_blocking=*/true);

            // Volteo Horizontal Aleatorio (50%)
            if (torch::rand({ 1 }).item<float>() > 0.5) {
                x_train = torch::flip(x_train, { 3 }); // Dim 3 = Ancho
                y_train = torch::flip(y_train, { 3 });
            }
            // Volteo Vertical Aleatorio (50%)
            if (torch::rand({ 1 }).item<float>() > 0.5) {
                x_train = torch::flip(x_train, { 2 }); // Dim 2 = Alto
                y_train = torch::flip(y_train, { 2 });
            }
            // Rotación Aleatoria (0, 90, 180, 270 grados)
            int k = torch::randint(0, 4, { 1 }).item<int>();
            if (k > 0) {
                x_train = torch::rot90(x_train, k, { 2, 3 });
                y_train = torch::rot90(y_train, k, { 2, 3 });
            }

            torch::Tensor predict = model->forward(x_train);
            auto loss = focal_loss(predict, y_train, 0.25, 2.0) + dice_loss(predict, y_train); // + criterion(predict, y_train)
            loss = loss / pasos_acumulacion;
            
            loss.backward();
            vueltas++;
            if (vueltas % pasos_acumulacion == 0 || vueltas == total_lotes) {
                optimizador.step();
                model->zero_grad();
            }

            lossAcumulada += loss.item<float>() * pasos_acumulacion;
            

            float porcentaje = (static_cast<float>(vueltas) / total_lotes) * 100.0f;
            float loss_actual_promedio = lossAcumulada / vueltas;

            std::cout << "\r   ↳ Época [" << i << "/" << ciclos << "] | Progreso: "
                << std::fixed << std::setprecision(1) << porcentaje << "% "
                << "| Loss actual: " << std::setprecision(4) << loss_actual_promedio
                << " (" << vueltas << "/" << total_lotes << " lotes)" << std::flush;
        }

        // Al terminar la época, imprimimos un salto de línea definitivo para guardar el resumen
        if (vueltas > 0) {
            std::cout << "\n➔ Época [" << i << "/" << ciclos << "] COMPLETADA - Pérdida Final: "
                << (lossAcumulada / vueltas) << "\n" << std::endl;
        }
    }

    torch::save(model, "unet_filamentos_solares.pt");

    // =====================================================================
    // INICIO DE INFERENCIA Y EXPORTACIÓN KAGGLE (DIRECTO EN MAIN)
    // =====================================================================

    std::cout << "\n🔍 Iniciando fase de inferencia en imágenes de test..." << std::endl;

    // Rutas para la inferencia
    std::string test_dir = base_path + "/MAGFiLO_1.0_Kaggle_2026/test/test_images";
    std::string output_dir = "/content/"; 

    fs::create_directories(output_dir);
    
    std::string output_csv = output_dir + "/submission.csv";
    std::ofstream csv_file(output_csv);
    csv_file << "filament_id,segmentation_rle\n";

    model->eval();
    torch::NoGradGuard no_grad;

    int procesadas = 0;

    for (const auto& entry : fs::directory_iterator(test_dir)) {
        if (!entry.is_regular_file()) continue;

        std::string filepath = entry.path().string();
        std::string filename = entry.path().filename().string();
        std::string base_name = entry.path().stem().string(); // Extraer el nombre de la imagen SIN la extensión (Ej: "20150125172714Mh")

        cv::Mat img_orig = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
        if (img_orig.empty()) continue;

        cv::Rect roi(119, 119, 1810, 1810);
        cv::Mat img_cropped = img_orig(roi);

        cv::Mat circular_mask = cv::Mat::zeros(img_cropped.size(), CV_8UC1);
        cv::Point center(img_cropped.cols / 2, img_cropped.rows / 2);
        cv::circle(circular_mask, center, img_cropped.cols / 2, cv::Scalar(255), -1);
        cv::bitwise_and(img_cropped, circular_mask, img_cropped);

        //////////////////////////////////////////////
        int step = 256;

        cv::Mat probs_sum = cv::Mat::zeros(1810, 1810, CV_32FC1);
        cv::Mat pixel_counts = cv::Mat::zeros(1810, 1810, CV_32FC1);

        for (int y = 0; y < 1810; y += step) {
            if (y + TARGET_SIZE > 1810) y = 1810 - TARGET_SIZE;
            for (int x = 0; x < 1810; x += step) {
                if (x + TARGET_SIZE > 1810) x = 1810 - TARGET_SIZE;

                cv::Rect patch_roi(x, y, TARGET_SIZE, TARGET_SIZE);
                cv::Mat patch = img_cropped(patch_roi).clone();

                torch::Tensor img_tensor = torch::from_blob(patch.data, { TARGET_SIZE, TARGET_SIZE, 1 }, torch::kByte).clone();
                img_tensor = img_tensor.permute({ 2, 0, 1 }).to(torch::kFloat32).div(255.0);
                img_tensor = img_tensor.unsqueeze(0).to(device);

                torch::Tensor logits = model->forward(img_tensor);
                torch::Tensor probs = torch::sigmoid(logits).squeeze().cpu().to(torch::kFloat32).contiguous();

                cv::Mat probs_mat(TARGET_SIZE, TARGET_SIZE, CV_32FC1, probs.data_ptr<float>());

                cv::add(probs_sum(patch_roi), probs_mat, probs_sum(patch_roi));
                cv::add(pixel_counts(patch_roi), cv::Scalar(1.0f), pixel_counts(patch_roi));
                
                if (x == 1810 - TARGET_SIZE) break;
            
            }
            if (y == 1810 - TARGET_SIZE) break;
        }

        ///////////////////////////////////////
        // Preprocesar (Redimensionar como en el entrenamiento)
        

        cv::Mat final_probs_1810;
        cv::divide(probs_sum, pixel_counts, final_probs_1810);

        cv::Mat pred_mask_1810_float, pred_mask_1810;
        cv::threshold(final_probs_1810, pred_mask_1810_float, 0.5, 255.0, cv::THRESH_BINARY);
        pred_mask_1810_float.convertTo(pred_mask_1810, CV_8UC1);

        cv::Mat pred_mask_2048 = cv::Mat::zeros(2048, 2048, CV_8UC1);
        pred_mask_1810.copyTo(pred_mask_2048(roi));

        // Opcional: Guardar la predicción completa para visualizar
        cv::imwrite(output_dir + "/pred_" + filename, pred_mask_2048);

        // 2. Separar los filamentos usando Connected Components
        cv::Mat labels, stats, centroids;
        // Devuelve el número de etiquetas (el fondo cuenta como la etiqueta 0)
        int num_labels = cv::connectedComponentsWithStats(pred_mask_2048, labels, stats, centroids, 8, CV_32S);
        
        const int MIN_PIXELS = 200;
        // 3. Iterar sobre cada filamento encontrado (empezamos en 1 para ignorar el fondo)
        for (int i = 1; i < num_labels; ++i) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < MIN_PIXELS) {
                continue;
            }
            // Crear una máscara donde SOLO este filamento es 255 y lo demás 0
            cv::Mat filament_mask = (labels == i);
            // Convertir a RLE usando tu función
            std::string rle_counts = maskToCOCORLE(filament_mask);
            // Crear el ID único (Ej: 20150125172714Mh_1)
            std::string filament_id = base_name + "_" + std::to_string(i);
            // Escribir en el CSV (sin comillas alrededor del RLE, como pide Kaggle)
            csv_file << filament_id << "," << rle_counts << "\n";
        }

        procesadas++;
        std::cout << "\r   ↳ Procesando imágenes test: " << procesadas << std::flush;
    }

    csv_file.close();
    std::cout << "\n Archivo listo para subir a Kaggle     " << output_csv << std::endl;

    
    return 0;


}
