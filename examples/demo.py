#!/usr/bin/env python3
"""
Demo script showcasing AI Outlier Detection Pipeline capabilities.

This script demonstrates:
1. Basic outlier detection on custom texts
2. File-based analysis
3. Full pipeline execution
4. Visualization generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import AIOutlierDetectionPipeline
import pandas as pd
import json

def demo_custom_text_analysis():
    """Demonstrate outlier detection on custom texts."""
    print("🔍 Demo 1: Custom Text Analysis")
    print("=" * 50)
    
    # Sample texts with clear outliers
    texts = [
        "Machine learning algorithms are powerful tools for data analysis",
        "Deep neural networks can learn complex patterns in data",
        "Artificial intelligence is transforming many industries",
        "My grandmother's apple pie recipe uses cinnamon and nutmeg",  # Outlier
        "Computer vision enables machines to interpret visual information",
        "The weather today is sunny with a chance of rain later",  # Outlier
        "Natural language processing helps computers understand human language"
    ]
    
    categories = ["tech"] * len(texts)
    
    try:
        pipeline = AIOutlierDetectionPipeline()
        results = pipeline.detect_outliers_in_text(texts, categories)
        
        print(f"📊 Analyzed {len(texts)} documents")
        print("\nResults by method:")
        
        for method, data in results.items():
            print(f"\n{method}:")
            print(f"  Outliers detected: {data['outlier_count']}")
            if data['outlier_indices']:
                print(f"  Outlier indices: {data['outlier_indices']}")
                print("  Outlier texts:")
                for idx in data['outlier_indices'][:3]:  # Show first 3
                    print(f"    [{idx}]: {texts[idx][:80]}...")
        
        print("\n✅ Custom text analysis completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_file_analysis():
    """Demonstrate analysis of a CSV file."""
    print("\n🔍 Demo 2: File-based Analysis")
    print("=" * 50)
    
    # Create sample CSV file
    sample_data = pd.DataFrame({
        'text': [
            "Quantum computing uses quantum mechanics principles",
            "Blockchain technology enables decentralized systems",
            "Cloud computing provides scalable infrastructure",
            "I love chocolate ice cream on summer days",  # Outlier
            "Cybersecurity protects against digital threats",
            "The cat sat on the mat and purred loudly",  # Outlier
            "Data science combines statistics and programming"
        ],
        'category': ['tech', 'tech', 'tech', 'personal', 'tech', 'personal', 'tech'],
        'id': range(1, 8)
    })
    
    # Save to temporary file
    csv_path = "temp_sample.csv"
    sample_data.to_csv(csv_path, index=False)
    
    try:
        pipeline = AIOutlierDetectionPipeline()
        
        # Load and analyze file
        df = pd.read_csv(csv_path)
        texts = df['text'].tolist()
        categories = df['category'].tolist()
        
        results = pipeline.detect_outliers_in_text(texts, categories)
        
        print(f"📄 Loaded file: {csv_path}")
        print(f"📊 Analyzed {len(texts)} documents")
        
        print("\nFile contents preview:")
        for i, (text, cat) in enumerate(zip(texts[:3], categories[:3])):
            print(f"  [{i}] ({cat}): {text[:60]}...")
        
        print("\nOutlier detection results:")
        for method, data in results.items():
            print(f"\n{method}: {data['outlier_count']} outliers")
            if data['outlier_indices']:
                for idx in data['outlier_indices'][:2]:
                    print(f"  Outlier [{idx}]: {texts[idx][:60]}...")
        
        print("\n✅ File analysis completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean up
        if os.path.exists(csv_path):
            os.remove(csv_path)

def demo_full_pipeline():
    """Demonstrate the complete pipeline on 20 Newsgroups data."""
    print("\n🔍 Demo 3: Full Pipeline Execution")
    print("=" * 50)
    
    try:
        pipeline = AIOutlierDetectionPipeline()
        
        print("🚀 Running complete pipeline on 20 Newsgroups dataset...")
        print("This may take a few minutes...")
        
        results = pipeline.run_full_pipeline(
            save_results=True,
            output_dir="demo_results"
        )
        
        print(f"\n📊 Pipeline Results:")
        print(f"Total documents processed: {results['total_documents']}")
        
        print("\nOutlier detection summary:")
        for method, count in results['summary_stats'].items():
            percentage = (count / results['total_documents']) * 100
            print(f"  {method}: {count} outliers ({percentage:.1f}%)")
        
        print(f"\n💾 Results saved to: demo_results/")
        print("📈 Visualizations and explanations generated!")
        
        # Show sample explanations
        if results['explanations']:
            print("\n🤖 Sample AI Explanations:")
            for method, explanations in list(results['explanations'].items())[:1]:
                print(f"\n{method} outlier explanation:")
                if explanations:
                    print(f"  {list(explanations.values())[0][:200]}...")
        
        print("\n✅ Full pipeline completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_api_usage():
    """Show how to use the API programmatically."""
    print("\n🔍 Demo 4: API Usage Example")
    print("=" * 50)
    
    print("To use the FastAPI interface:")
    print("1. Start the server: python api.py")
    print("2. Visit: http://localhost:8000/docs")
    print("3. Or use curl:")
    
    example_curl = '''
curl -X POST "http://localhost:8000/analyze/texts" \\
     -H "Content-Type: application/json" \\
     -d '{
       "texts": [
         "Machine learning is fascinating",
         "I love pizza and ice cream"
       ],
       "categories": ["tech", "food"]
     }'
    '''
    print(example_curl)

def main():
    """Run all demos."""
    print("🚀 AI Outlier Detection Pipeline - Demo Suite")
    print("=" * 60)
    
    # Check if API key is configured
    if not os.getenv('NEBIUS_API_KEY'):
        print("⚠️  Warning: NEBIUS_API_KEY not found in environment")
        print("Please set your API key in .env file or environment variables")
        print("Some demos may fail without proper API configuration")
        print()
    
    try:
        demo_custom_text_analysis()
        demo_file_analysis()
        
        # Ask user if they want to run the full pipeline (takes longer)
        response = input("\n🤔 Run full pipeline demo? This takes 2-3 minutes (y/N): ")
        if response.lower() in ['y', 'yes']:
            demo_full_pipeline()
        else:
            print("⏭️  Skipping full pipeline demo")
        
        demo_api_usage()
        
        print("\n🎉 All demos completed successfully!")
        print("\nNext steps:")
        print("- Try: python cli.py --help")
        print("- Start API: python api.py")
        print("- Run tests: make test")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")

if __name__ == "__main__":
    main()