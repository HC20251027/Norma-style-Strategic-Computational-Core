"""
图像处理模块
支持图像预处理、格式转换、压缩和特征提取
"""

import asyncio
import base64
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from PIL import Image, ImageEnhance, ImageFilter
import io
import logging

from ..utils import format_file_size

logger = logging.getLogger(__name__)

@dataclass
class ImageProcessingConfig:
    """图像处理配置"""
    max_width: int = 2048
    max_height: int = 2048
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    supported_formats: List[str] = None
    quality: int = 85
    enable_compression: bool = True
    enable_resize: bool = True
    enable_enhancement: bool = False
    target_format: str = "JPEG"  # JPEG, PNG, WEBP
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ["JPEG", "PNG", "WEBP", "BMP", "TIFF"]

@dataclass
class ImageMetadata:
    """图像元数据"""
    width: int
    height: int
    format: str
    mode: str
    file_size: int
    has_transparency: bool
    color_count: int
    dpi: Optional[Tuple[int, int]] = None
    exif_data: Optional[Dict[str, Any]] = None

class ImageProcessor:
    """图像处理器"""
    
    def __init__(self, config: ImageProcessingConfig = None):
        self.config = config or ImageProcessingConfig()
        self.supported_mime_types = {
            'JPEG': 'image/jpeg',
            'PNG': 'image/png',
            'WEBP': 'image/webp',
            'BMP': 'image/bmp',
            'TIFF': 'image/tiff'
        }
    
    async def process_image(
        self,
        image_data: Union[bytes, str, io.BytesIO],
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理图像
        
        Args:
            image_data: 图像数据（字节串、base64字符串或BytesIO对象）
            **kwargs: 额外参数
            
        Returns:
            处理后的图像和元数据
        """
        try:
            # 1. 解析图像数据
            pil_image, original_size = await self._parse_image_data(image_data)
            
            # 2. 验证图像
            await self._validate_image(pil_image)
            
            # 3. 提取元数据
            metadata = await self._extract_metadata(pil_image, original_size)
            
            # 4. 预处理图像
            processed_image = await self._preprocess_image(pil_image)
            
            # 5. 转换格式和压缩
            final_image, final_size = await self._convert_and_compress(processed_image)
            
            # 6. 生成图像哈希
            image_hash = await self._generate_image_hash(final_image)
            
            return {
                'image': final_image,
                'metadata': metadata,
                'original_size': original_size,
                'processed_size': final_size,
                'compression_ratio': original_size / final_size if final_size > 0 else 1,
                'hash': image_hash,
                'processing_info': {
                    'resized': processed_image.size != pil_image.size,
                    'compressed': final_size < original_size,
                    'format_changed': metadata.format != self.config.target_format
                }
            }
            
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            raise
    
    async def _parse_image_data(self, image_data: Union[bytes, str, io.BytesIO]) -> Tuple[Image.Image, int]:
        """解析图像数据"""
        if isinstance(image_data, bytes):
            pil_image = Image.open(io.BytesIO(image_data))
            original_size = len(image_data)
        elif isinstance(image_data, str):
            # 假设是base64编码
            if image_data.startswith('data:image'):
                # 移除data URL前缀
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            pil_image = Image.open(io.BytesIO(image_bytes))
            original_size = len(image_bytes)
        elif isinstance(image_data, io.BytesIO):
            pil_image = Image.open(image_data)
            original_size = image_data.tell()
        else:
            raise ValueError("不支持的图像数据格式")
        
        return pil_image, original_size
    
    async def _validate_image(self, pil_image: Image.Image):
        """验证图像"""
        # 检查格式支持
        if pil_image.format not in self.config.supported_formats:
            raise ValueError(f"不支持的图像格式: {pil_image.format}")
        
        # 检查文件大小（如果可获取）
        # 注意：PIL Image对象本身不包含文件大小信息
        
        # 检查图像尺寸
        if pil_image.width == 0 or pil_image.height == 0:
            raise ValueError("图像尺寸无效")
        
        # 检查超大图像
        if pil_image.width > 10000 or pil_image.height > 10000:
            raise ValueError("图像尺寸过大")
    
    async def _extract_metadata(self, pil_image: Image.Image, file_size: int) -> ImageMetadata:
        """提取图像元数据"""
        # 检查透明度
        has_transparency = False
        if pil_image.mode in ('RGBA', 'LA', 'P'):
            has_transparency = True
        
        # 颜色数量
        color_count = len(pil_image.getcolors()) if pil_image.getcolors() else 0
        
        # DPI信息（如果有）
        dpi = None
        try:
            dpi = pil_image.info.get('dpi')
        except:
            pass
        
        # EXIF数据（如果有）
        exif_data = None
        try:
            exif_dict = {}
            if hasattr(pil_image, '_getexif') and pil_image._getexif():
                exif = pil_image._getexif()
                for tag_id, value in exif.items():
                    tag = Image.ExifTags.TAGS.get(tag_id, tag_id)
                    exif_dict[tag] = value
            exif_data = exif_dict
        except:
            pass
        
        return ImageMetadata(
            width=pil_image.width,
            height=pil_image.height,
            format=pil_image.format or 'UNKNOWN',
            mode=pil_image.mode,
            file_size=file_size,
            has_transparency=has_transparency,
            color_count=color_count,
            dpi=dpi,
            exif_data=exif_data
        )
    
    async def _preprocess_image(self, pil_image: Image.Image) -> Image.Image:
        """预处理图像"""
        processed_image = pil_image.copy()
        
        # 调整大小
        if self.config.enable_resize:
            processed_image = await self._resize_image(processed_image)
        
        # 图像增强
        if self.config.enable_enhancement:
            processed_image = await self._enhance_image(processed_image)
        
        return processed_image
    
    async def _resize_image(self, pil_image: Image.Image) -> Image.Image:
        """调整图像大小"""
        if (pil_image.width <= self.config.max_width and 
            pil_image.height <= self.config.max_height):
            return pil_image
        
        # 计算新尺寸
        width_ratio = self.config.max_width / pil_image.width
        height_ratio = self.config.max_height / pil_image.height
        ratio = min(width_ratio, height_ratio)
        
        new_width = int(pil_image.width * ratio)
        new_height = int(pil_image.height * ratio)
        
        # 调整大小
        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        logger.info(f"图像大小调整: {pil_image.width}x{pil_image.height} -> {new_width}x{new_height}")
        
        return resized_image
    
    async def _enhance_image(self, pil_image: Image.Image) -> Image.Image:
        """图像增强"""
        enhanced = pil_image.copy()
        
        # 增强对比度
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.1)  # 略微增强对比度
        
        # 增强锐度
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.1)  # 略微增强锐度
        
        # 轻微模糊以减少噪点
        enhanced = enhanced.filter(ImageFilter.SMOOTH_MORE)
        
        return enhanced
    
    async def _convert_and_compress(self, pil_image: Image.Image) -> Tuple[bytes, int]:
        """转换格式和压缩"""
        output_format = self.config.target_format
        
        # 处理透明通道
        if output_format == 'JPEG' and pil_image.mode in ('RGBA', 'LA', 'P'):
            # JPEG不支持透明度，转换为RGB并添加白色背景
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            if pil_image.mode == 'P':
                pil_image = pil_image.convert('RGBA')
            background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode == 'RGBA' else None)
            pil_image = background
        elif pil_image.mode not in ('RGB', 'RGBA', 'L', 'LA'):
            # 转换为支持的模式
            pil_image = pil_image.convert('RGB')
        
        # 压缩并保存
        buffer = io.BytesIO()
        
        save_kwargs = {
            'format': output_format,
            'quality': self.config.quality,
            'optimize': True
        }
        
        # 特殊格式参数
        if output_format == 'WEBP':
            save_kwargs['method'] = 6  # 最佳压缩
        elif output_format == 'PNG':
            save_kwargs['optimize'] = True
            save_kwargs['compress_level'] = 9  # 最高压缩级别
        
        pil_image.save(buffer, **save_kwargs)
        
        compressed_data = buffer.getvalue()
        compressed_size = len(compressed_data)
        
        logger.info(f"图像压缩: {format_file_size(self._estimate_uncompressed_size(pil_image))} -> {format_file_size(compressed_size)}")
        
        return compressed_data, compressed_size
    
    def _estimate_uncompressed_size(self, pil_image: Image.Image) -> int:
        """估算未压缩大小"""
        # 估算RGB图像的未压缩大小
        bytes_per_pixel = {
            'L': 1,
            'RGB': 3,
            'RGBA': 4,
            'P': 1  # 调色板模式
        }
        
        channels = bytes_per_pixel.get(pil_image.mode, 3)
        return pil_image.width * pil_image.height * channels
    
    async def _generate_image_hash(self, image_data: bytes) -> str:
        """生成图像哈希"""
        return hashlib.md5(image_data).hexdigest()
    
    async def batch_process(
        self,
        images: List[Union[bytes, str, io.BytesIO]]
    ) -> List[Dict[str, Any]]:
        """批量处理图像"""
        tasks = [self.process_image(image) for image in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'image': None,
                    'error': str(result),
                    'metadata': None,
                    'original_size': 0,
                    'processed_size': 0
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def extract_features(self, pil_image: Image.Image) -> Dict[str, Any]:
        """提取图像特征"""
        features = {}
        
        # 基础统计信息
        if pil_image.mode == 'RGB':
            # 计算颜色直方图
            hist_r = pil_image.histogram()[0:256]
            hist_g = pil_image.histogram()[256:512]
            hist_b = pil_image.histogram()[512:768]
            
            features['color_histogram'] = {
                'red': hist_r,
                'green': hist_g,
                'blue': hist_b
            }
            
            # 计算平均颜色
            avg_r = sum(i * hist_r[i] for i in range(256)) / sum(hist_r)
            avg_g = sum(i * hist_g[i] for i in range(256)) / sum(hist_g)
            avg_b = sum(i * hist_b[i] for i in range(256)) / sum(hist_b)
            
            features['average_color'] = (avg_r, avg_g, avg_b)
        
        # 图像尺寸特征
        features['aspect_ratio'] = pil_image.width / pil_image.height
        features['pixel_density'] = pil_image.width * pil_image.height
        
        # 边缘检测（简化）
        try:
            edges = pil_image.filter(ImageFilter.FIND_EDGES)
            edge_pixels = sum(edges.histogram())
            features['edge_density'] = edge_pixels / (pil_image.width * pil_image.height * 255)
        except:
            features['edge_density'] = 0
        
        return features
    
    async def detect_objects(self, pil_image: Image.Image) -> List[Dict[str, Any]]:
        """检测图像中的对象（占位符）"""
        # 这里应该集成对象检测模型
        # 简化实现：返回模拟结果
        return [
            {
                'class': 'object',
                'confidence': 0.8,
                'bbox': [0.1, 0.1, 0.9, 0.9]
            }
        ]
    
    async def generate_thumbnail(
        self,
        pil_image: Image.Image,
        size: Tuple[int, int] = (128, 128)
    ) -> bytes:
        """生成缩略图"""
        thumbnail = pil_image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        thumbnail.save(buffer, format='JPEG', quality=85)
        
        return buffer.getvalue()
    
    def image_to_base64(self, image_data: bytes, format: str = 'JPEG') -> str:
        """将图像转换为base64"""
        return base64.b64encode(image_data).decode('utf-8')
    
    def base64_to_image(self, base64_string: str) -> bytes:
        """将base64转换为图像"""
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        return base64.b64decode(base64_string)
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        logger.info(f"图像处理配置已更新: {kwargs}")
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的格式"""
        return self.config.supported_formats.copy()
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.__dict__.copy()