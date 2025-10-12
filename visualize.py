#!/usr/bin/env python3
"""
Visualization Module
Display search results with images
"""

import matplotlib.pyplot as plt
from PIL import Image
import os


def display_search_results(results, query, output_dir=None, show_plot=True):
    """
    Display search results with images
    
    Args:
        results: List of search results
        query: Original query text
        output_dir: Directory to save visualization (optional)
        show_plot: Whether to display plot
    """
    
    if not results:
        print("No results to display")
        return
    
    print(f"\n Top {len(results)} matches for: '{query}'\n")
    
    # Text summary
    for i, result in enumerate(results, 1):
        person_id = result['person_id']
        max_score = result['max_score']
        avg_score = result['avg_score']
        total_images = result['total_images']
        
        print(f"{i}. Person ID: {person_id}")
        print(f"Best Score: {max_score:.3f}")
        print(f"Avg Score: {avg_score:.3f}")
        print(f"Total Images: {total_images}")
        
        # Confidence level
        if max_score > 0.8:
            confidence = "EXCELLENT"
        elif max_score > 0.7:
            confidence = "VERY HIGH"
        elif max_score > 0.6:
            confidence = "HIGH"
        elif max_score > 0.5:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        print(f"Confidence: {confidence}\n")
    
    # Visual display
    n_results = len(results)
    n_cols = 3  # Show 3 images per person
    
    fig, axes = plt.subplots(n_results, n_cols, figsize=(12, 4*n_results))
    
    if n_results == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        person_id = result['person_id']
        sample_images = result['sample_images']
        max_score = result['max_score']
        
        for j in range(n_cols):
            ax = axes[i, j]
            
            if j < len(sample_images):
                try:
                    img = Image.open(sample_images[j]).convert("RGB")
                    ax.imshow(img)
                    
                    # Extract frame number from filename
                    frame_num = os.path.basename(sample_images[j]).split('.')[0]
                    ax.set_title(f'Person {person_id}\n{frame_num}\nScore: {max_score:.3f}', 
                               fontsize=9)
                except Exception as e:
                    ax.text(0.5, 0.5, 'Error\nloading\nimage', 
                           ha='center', va='center')
                    ax.set_title(f'Person {person_id}')
            else:
                ax.axis('off')
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle(f'Search Results: "{query}"', fontsize=16, y=0.995)
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_'))
        save_path = f"{output_dir}/search_{safe_query.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def display_person_details(person_id, data, max_images=12):
    """
    Display detailed view of a specific person
    
    Args:
        person_id: Person ID to display
        data: Complete data structure
        max_images: Maximum images to display
    """
    
    person_images = data['person_images']
    
    if person_id not in person_images:
        print(f"Person {person_id} not found")
        return
    
    images = person_images[person_id][:max_images]
    
    print(f"\n Person ID: {person_id}")
    print(f"Total images: {len(person_images[person_id])}")
    print(f"Showing: {len(images)} images\n")
    
    # Display in grid
    n_cols = 4
    n_rows = (len(images) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            frame_num = os.path.basename(img_path).split('.')[0]
            ax.set_title(frame_num, fontsize=9)
        except Exception:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(len(images), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')
    
    plt.suptitle(f'Person {person_id} - All Appearances', fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    """Standalone testing"""
    print("Visualization module - import to use functions")


if __name__ == '__main__':
    main()
