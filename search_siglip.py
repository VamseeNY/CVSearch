#!/usr/bin/env python3
"""
SigLIP Search Module
Vision-language model for semantic person search
"""

import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import numpy as np


class SigLIPSearchEngine:
    """SigLIP-based person search engine"""
    
    def __init__(self, model_name="google/siglip-so400m-patch14-384"):
        """
        Initialize SigLIP search engine
        
        Args:
            model_name: SigLIP model name from HuggingFace
        """
        
        print("Loading SigLIP - Google's Vision-Language Model")
        print(f"Model: {model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print("SigLIP model loaded successfully!")
    
    def compute_similarity(self, image_path, text_query):
        """
        Compute image-text similarity using SigLIP
        
        Args:
            image_path: Path to image file
            text_query: Text description
        
        Returns:
            float: Similarity score (0-1)
        """
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Process inputs
            inputs = self.processor(
                text=[text_query],
                images=image,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)
            
            # Compute similarity
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # SigLIP uses sigmoid activation
            logits = outputs.logits_per_image
            similarity = torch.sigmoid(logits[0, 0]).item()
            
            return similarity
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return 0.0
    
    def compute_similarity_enhanced(self, image_path, text_query):
        """
        Enhanced similarity with multiple text variations
        
        Args:
            image_path: Path to image
            text_query: Base text query
        
        Returns:
            float: Maximum similarity across variations
        """
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Create text variations
            text_variations = [
                text_query,
                f"a photo of {text_query}",
                f"an image showing {text_query}",
                f"a picture of {text_query}",
                f"this shows {text_query}"
            ]
            
            # Process all variations
            inputs = self.processor(
                text=text_variations,
                images=[image] * len(text_variations),
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get diagonal similarities (each image with its corresponding text)
            logits = outputs.logits_per_image
            similarities = torch.sigmoid(torch.diag(logits))
            
            # Return best similarity
            return similarities.max().item()
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return 0.0
    
    def search_persons(self, person_images, query, top_k=5, 
                       max_images_per_person=10, use_enhanced=True):
        """
        Search for persons matching text query
        
        Args:
            person_images: Dict {person_id: [image_paths]}
            query: Text description
            top_k: Number of top results
            max_images_per_person: Max images to test per person
            use_enhanced: Use enhanced similarity with text variations
        
        Returns:
            list: Top K results with scores
        """
        
        print(f"Searching for: '{query}'")
        print(f"Searching through {len(person_images)} persons...")
        
        results = []
        similarity_func = self.compute_similarity_enhanced if use_enhanced else self.compute_similarity
        
        for person_id, image_paths in person_images.items():
            # Test up to max_images_per_person
            test_images = image_paths[:max_images_per_person]
            
            scores = []
            for img_path in test_images:
                score = similarity_func(img_path, query)
                scores.append(score)
            
            if scores:
                results.append({
                    'person_id': person_id,
                    'max_score': max(scores),
                    'avg_score': np.mean(scores),
                    'min_score': min(scores),
                    'scores': scores,
                    'sample_images': test_images[:6],  # Keep top 6 for display
                    'total_images': len(image_paths)
                })
        
        # Sort by max score
        results.sort(key=lambda x: x['max_score'], reverse=True)
        
        print(f"Found {len(results)} persons, returning top {min(top_k, len(results))}")
        
        return results[:top_k]


def main():
    """Standalone testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test SigLIP search')
    parser.add_argument('--image', required=True, help='Test image path')
    parser.add_argument('--query', required=True, help='Text query')
    
    args = parser.parse_args()
    
    # Initialize search engine
    engine = SigLIPSearchEngine()
    
    # Test similarity
    score = engine.compute_similarity(args.image, args.query)
    enhanced_score = engine.compute_similarity_enhanced(args.image, args.query)
    
    print(f"\nResults for: '{args.query}'")
    print(f"Basic similarity: {score:.3f}")
    print(f"Enhanced similarity: {enhanced_score:.3f}")


if __name__ == '__main__':
    main()
