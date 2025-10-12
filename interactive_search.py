#!/usr/bin/env python3
"""
Interactive Search Interface
Command-line interface for person search
"""

import argparse
import json
import os
from datetime import datetime

from load_data import load_phase1_results
from search_siglip import SigLIPSearchEngine
from visualize import display_search_results, display_person_details


class InteractiveSearch:
    """Interactive search interface"""
    
    def __init__(self, results_dir, output_dir=None):
        """
        Initialize interactive search
        
        Args:
            results_dir: Phase 1 results directory
            output_dir: Directory to save search results
        """
        
        print("=" * 70)
        print("INTERACTIVE PERSON SEARCH")
        print("=" * 70)
        print()
        
        # Load data
        self.data = load_phase1_results(results_dir)
        
        # Initialize search engine
        self.search_engine = SigLIPSearchEngine()
        
        # Setup output directory
        self.output_dir = output_dir or f"{results_dir}/search_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Search history
        self.search_history = []
        
        print()
        print("=" * 70)
        print("SYSTEM READY")
        print("=" * 70)
    
    def quick_search(self, query, top_k=5, show_images=True):
        """
        Quick search with text query
        
        Args:
            query: Text description
            top_k: Number of results
            show_images: Whether to display images
        
        Returns:
            list: Search results
        """
        
        results = self.search_engine.search_persons(
            self.data['person_images'],
            query,
            top_k=top_k,
            max_images_per_person=10,
            use_enhanced=True
        )
        
        if not results:
            print("No matches found")
            return []
        
        # Display results
        if show_images:
            display_search_results(results, query, self.output_dir)
        else:
            # Text-only results
            print(f"\n Results for: '{query}'")
            for i, result in enumerate(results, 1):
                score = result['max_score']
                person_id = result['person_id']
                confidence = "Very high confidence - " if score > 0.8 else "High Confidence - " if score > 0.7 else "Good confidence - " if score > 0.6 else "Confidence - "
                print(f"  {i}. Person {person_id}: {score:.3f} {confidence}")
        
        # Save to history
        self.search_history.append({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'num_results': len(results),
            'top_score': results[0]['max_score'] if results else 0
        })
        
        return results
    
    def run_interactive(self):
        """Run interactive search loop"""
        
        print("\n Example queries:")
        print("  • person wearing red shirt")
        print("  • man with backpack")
        print("  • woman in dark dress")
        print("  • person with hat")
        print("  • tall person")
        print("\n Commands:")
        print("  • 'quit' or 'exit' - Exit search")
        print("  • 'history' - Show search history")
        print("  • 'person <id>' - View specific person details")
        
        while True:
            print("\n" + "-" * 70)
            query = input(" Enter query (or command): ").strip()
            
            if not query:
                continue
            
            # Check for commands
            if query.lower() in ['quit', 'exit', 'q']:
                print(" Exiting search...")
                break
            
            elif query.lower() == 'history':
                self.show_history()
                continue
            
            elif query.lower().startswith('person '):
                try:
                    person_id = int(query.split()[1])
                    display_person_details(person_id, self.data)
                except (ValueError, IndexError):
                    print(" Invalid person ID. Usage: person <id>")
                continue
            
            # Perform search
            try:
                results = self.quick_search(query, top_k=3, show_images=False)
                
                if results:
                    show_details = input("\n Show detailed results with images? (y/n): ")
                    if show_details.lower().startswith('y'):
                        display_search_results(results, query, self.output_dir)
                
            except KeyboardInterrupt:
                print("\n Search interrupted")
                continue
            except Exception as e:
                print(f" Error: {e}")
                continue
        
        # Save history on exit
        self.save_history()
    
    def show_history(self):
        """Display search history"""
        
        if not self.search_history:
            print(" No search history yet")
            return
        
        print("\n Search History:")
        print("-" * 70)
        for i, entry in enumerate(self.search_history, 1):
            print(f"{i}. '{entry['query']}'")
            print(f"   Time: {entry['timestamp']}")
            print(f"   Results: {entry['num_results']}, Top Score: {entry['top_score']:.3f}")
    
    def save_history(self):
        """Save search history to file"""
        
        if not self.search_history:
            return
        
        history_file = f"{self.output_dir}/search_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.search_history, f, indent=2)
        
        print(f"\n Search history saved: {history_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Interactive person search using SigLIP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python interactive_search.py --results_dir ./multi_camera_results
  
  # Quick search
  python interactive_search.py --results_dir ./results --query "person in red shirt"
  
  # Quick search without images
  python interactive_search.py --results_dir ./results --query "man with backpack" --no_images
        """
    )
    
    parser.add_argument('--results_dir', required=True, 
                       help='Phase 1 results directory')
    parser.add_argument('--output_dir', 
                       help='Output directory for search results')
    parser.add_argument('--query', 
                       help='Quick search query (skips interactive mode)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top results')
    parser.add_argument('--no_images', action='store_true',
                       help='Text-only results (no images)')
    
    args = parser.parse_args()
    
    # Initialize search system
    search = InteractiveSearch(args.results_dir, args.output_dir)
    
    if args.query:
        # Quick search mode
        search.quick_search(args.query, top_k=args.top_k, show_images=not args.no_images)
    else:
        # Interactive mode
        search.run_interactive()
    
    print("\n Search complete!")


if __name__ == '__main__':
    main()
