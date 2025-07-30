import cv2
import torch
import torchvision.models as models
# import clip
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import numpy as np


# Base Wrapper Class
class VisionTransformerWrapper:
    def __init__(self, model_name, device, smaller_edge_size=224, half_precision=False):
        self.device = device
        self.smaller_edge_size = smaller_edge_size
        self.half_precision = half_precision
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        raise NotImplementedError("This method should be overridden in a subclass")
    
    def extract_features(self, img_tensor):
        raise NotImplementedError("This method should be overridden in a subclass")


# ViT-B/16 Wrapper
class ViTWrapper(VisionTransformerWrapper):
    def load_model(self):
        if self.model_name == "vit_b_16":
            model = models.vit_b_16(weights = models.ViT_B_16_Weights.DEFAULT)
            self.transform = models.ViT_B_16_Weights.DEFAULT.transforms()
            self.grid_size = (14,14)
        elif self.model_name == "vit_b_32":
            model = models.vit_b_32(weights = models.ViT_B_32_Weights.DEFAULT)
            self.transform = models.ViT_B_32_Weights.DEFAULT.transforms()
            self.grid_size = (7,7)
        elif self.model_name == "vit_l_16":
            model = models.vit_l_16(weights = models.ViT_L_16_Weights.DEFAULT)
            self.transform = models.ViT_L_16_Weights.DEFAULT.transforms()
            self.grid_size = (14,14)
        elif self.model_name == "vit_l_32":
            model = models.vit_l_32(weights = models.ViT_L_32_Weights.DEFAULT)
            self.transform = models.ViT_L_32_Weights.DEFAULT.transforms()
            self.grid_size = (7,7)
        else:
            raise ValueError(f"Unknown ViT model name: {self.model_name}")
        
        model.eval()
        # print(self.transform)

        return model.to(self.device)
    
    def prepare_image(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img_tensor = self.transform(img).unsqueeze(0)
        return img_tensor, self.grid_size

    def extract_features(self, img_tensor):
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            patches = self.model._process_input(img_tensor)
            class_token = self.model.class_token.expand(patches.size(0), -1, -1)
            patches = torch.cat((class_token, patches), dim=1)
            patch_features = self.model.encoder(patches)
            return patch_features[:, 1:, :].squeeze().cpu().numpy()  # Exclude the class token

    def get_embedding_visualization(self, tokens, grid_size = (14,14), resized_mask=None, normalize=True):
        pca = PCA(n_components=3, svd_solver='randomized')
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*self.grid_size, -1))
        if normalize:
            normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
            return normalized_tokens
        else:
            return reduced_tokens

    def compute_background_mask(self, img_features, grid_size, threshold = 10, masking_type = False):
        # No masking for ViT supported at the moment... (Only DINOv2)
        return np.ones(img_features.shape[0], dtype=bool)
    

# DINOv2 Wrapper
class DINOv2Wrapper(VisionTransformerWrapper):
    def load_model(self):
        model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        model.eval()

        # print(f"Loaded model: {self.model_name}")
        # print("Resizing images to", self.smaller_edge_size)

        # Set transform for DINOv2
        if self.smaller_edge_size > 0:
            print(f"Do resize the image to {self.smaller_edge_size}")
            self.transform = transforms.Compose([
                transforms.Resize(size=self.smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
                ])
        else:
            # Do not resize the image; operate on the original resolution
            print("Do not resize the image")
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # imagenet defaults
            ])
        
        return model.to(self.device)
    
    def prepare_image(self, img):
        '''
        if isinstance(img, str):
        這行代碼檢查傳入的 img 是否是 字符串（str），即圖像的文件路徑。
        如果 img 是字符串，則使用 PIL.Image.open() 打開圖像文件，並將其轉換為 RGB 模式（.convert("RGB")），確保圖像是 3 通道的（即每個像素有紅、綠、藍三個顏色通道）。
        img 是圖像的文件路徑，這行代碼將其加載為圖像對象。

        elif isinstance(img, np.ndarray):
        這行代碼檢查傳入的 img 是否是 NumPy 陣列（np.ndarray），即圖像數據已經是數組形式。
        如果 img 是 NumPy 陣列，則使用 PIL.Image.fromarray(img) 將其轉換為 PIL.Image 對象。這樣可以進行後續的圖像處理。
        
        image_tensor = self.transform(img)
        這行代碼將圖像（無論是從文件加載的，還是從 NumPy 陣列轉換的）通過 self.transform 進行預處理。
        self.transform 是在初始化模型時定義的一個圖像轉換操作（例如：縮放、裁剪、標準化等）。
        這通常是基於 torchvision.transforms 模塊的操作，會將圖像轉換為 Tensor 格式並進行標準化處理，以便可以輸入到神經網絡中。
        image_tensor 是處理後的圖像，這樣它已經準備好進行模型的推理。
        
        height, width = image_tensor.shape[1:]
        行代碼獲取圖像的 高度（height）和 寬度（width）。
        
        cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size
        這行代碼是用來裁剪圖像，使其尺寸可以被 patch_size 整除。為什麼需要這麼做呢？
        因為 Vision Transformer（ViT）這類模型通常將圖像切割成固定大小的 patch 來進行處理，這樣可以提高模型的效率和性能。
        
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]
        這行代碼是用來只保留裁剪後的 tensor
        
        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
        這行代碼計算圖像在每個維度上可以分割成多少個 patch。
        '''
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        image_tensor = self.transform(img)
        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[1:] # C x H x W
        cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
        return image_tensor, grid_size
    

    def extract_features(self, image_tensor):
        '''
        extract_features 方法負責從處理過的圖像中提取特徵。
        這個方法會將圖像轉換為適當的大小和形狀，然後通過 DINOv2 模型的中間層來提取特徵。
        具體來說，它會返回圖像的 tokens（模型的中間層輸出）。
        這些 tokens 是圖像的高級表示，這些表示可以用來進行後續的處理，如異常檢測等。
        '''
        '''
        if self.half_precision:
        這部分代碼的目的是處理圖像張量並將其傳送到指定設備（如 GPU）上，並根據 half_precision 標誌選擇是否使用 半精度浮點數（float16）。
        unsqueeze(0) 是 PyTorch 中用來增加一個維度的操作，將 image_tensor 的形狀從 (C, H, W) 變為 (1, C, H, W)，
        這是因為模型期望的輸入是批次形式，即 (batch_size, channels, height, width)，即使是單張圖像也需要增加批次維度。
        
        tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
        self.model.get_intermediate_layers(image_batch)：
        這行代碼調用了模型的 get_intermediate_layers 方法，將 image_batch（處理過的圖像）傳入模型。該方法返回模型的中間層輸出，這些中間層的特徵表示即為 tokens。
        具體來說，這是一個從模型的中間層提取特徵的操作，這些中間層的輸出反映了圖像的高級語義特徵，這些特徵可以用來進行後續的處理，如異常檢測或圖像分類。
        [0]：
        get_intermediate_layers 方法返回的是一個列表，這裡選取了列表中的第一個元素，即模型中某個特定層的輸出。這個選取是根據模型的設計而定的，通常是深度學習模型中的某一層特徵。
        squeeze()：
        squeeze() 是一個 PyTorch 操作，用來去除多餘的維度。這裡，squeeze() 用來移除維度為 1 的軸。
        假設模型的輸出形狀是 (1, num_tokens, feature_dim)，squeeze() 會將其轉換為 (num_tokens, feature_dim)，這樣返回的 tokens 就是圖像的特徵表示，包含每個 patch 的高級特徵。
        '''
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half().to(self.device)
            else:
                image_batch = image_tensor.unsqueeze(0).to(self.device)

            # 這裡的 tokens 代表的是圖片被切成 patch 後，並丟入 encoder 所提取的特徵
            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
            
            # print(f"tokens:{len(tokens)}")
        return tokens.cpu().numpy()


    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None, normalize=True):
        '''
        這段代碼是用來將提取的 tokens（特徵）可視化的，並且使用 PCA（主成分分析） 降維，
        將高維特徵映射到三維空間。具體來說，它將每個 token（圖像的每個 patch）投影到 3D 空間中，並根據需要進行正規化處理。
        
        pca = PCA(n_components=3, svd_solver='randomized'):
        PCA（主成分分析） 是一種常見的降維技術，用於將高維數據映射到較低維度的空間。在這裡，我們希望將原本的高維 tokens 降到 3 維，這樣可以進行 3D 可視化。
        n_components=3：這表示我們希望將 tokens 降維到 3 維空間。
        svd_solver='randomized'：這是用來解決 PCA 的奇異值分解（SVD）的計算方式。'randomized' 方法通常在數據量大時速度較快。
        
        if resized_mask is not None:
        這行代碼檢查 resized_mask 是否為 None。resized_mask 是一個布爾值掩碼，它可以用來選擇特定的 tokens。如果 resized_mask 不為 None，
        則將 tokens 根據這個掩碼進行篩選，選擇需要的 tokens。
        例如，resized_mask 可能用來選擇前景區域的 tokens，忽略背景區域。
        
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1)):
        這行代碼會將 reduced_tokens 重新塑形為符合網格大小（grid_size）的形狀。grid_size 通常表示圖像中每個方向上的 patch 數量（例如 14x14 或 7x7），
        因此，這裡的 -1 表示保持其他維度不變，根據數據自動計算剩餘的維度大小。
        
        '''
        pca = PCA(n_components=3, svd_solver='randomized')
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        if normalize:
            normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
            return normalized_tokens
        else:
            return reduced_tokens


    def compute_background_mask_from_image(self, image, threshold = 10, masking_type = None):
        '''
        compute_background_mask_from_image 和 compute_background_mask 方
        法負責計算背景遮罩。在異常檢測任務中，背景遮罩通常是用來標示那些被認為是「背景」的區域。
        這可以通過分析 tokens（模型的中間層輸出）來實現。
        '''
        image_tensor, grid_size = self.prepare_image(image)
        tokens = self.extract_features(image_tensor)
        return self.compute_background_mask(tokens, grid_size, threshold, masking_type)


    def compute_background_mask(self, img_features, grid_size, threshold = 10, masking_type = False, kernel_size = 3, border = 0.2):
        # Kernel size for morphological operations should be odd
        '''
        masking_type = False 代表全都是前景不會有背景 ，代表我要拿整張圖片，在建立 memory bank 就會這樣做
        compute_background_mask 方法會計算 PCA 中的第一主成分，
        然後根據閾值進行遮罩計算，這樣可以區分出背景和異常區域。
        '''
        pca = PCA(n_components=1, svd_solver='randomized')
        first_pc = pca.fit_transform(img_features.astype(np.float32))
        if masking_type == True:
            mask = first_pc > threshold
            # test whether the center crop of the images is kept (adaptive masking), adapt if your objects of interest are not centered!
            m = mask.reshape(grid_size)[int(grid_size[0] * border):int(grid_size[0] * (1-border)), int(grid_size[1] * border):int(grid_size[1] * (1-border))]
            if m.sum() <=  m.size * 0.35:
                mask = - first_pc > threshold
            # postprocess mask, fill small holes in the mask, enlarge slightly
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
        elif masking_type == False:
            mask = np.ones_like(first_pc, dtype=bool)
        return mask.squeeze()


def get_model(model_name, device, smaller_edge_size=448):
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print(f"Smaller edge size: {smaller_edge_size}")

    if model_name.startswith("vit"):
        return ViTWrapper(model_name, device, smaller_edge_size)
    elif model_name.startswith("dinov2"):
        return DINOv2Wrapper(model_name, device, smaller_edge_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")