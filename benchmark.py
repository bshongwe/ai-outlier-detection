#!/usr/bin/env python3
"""
Performance benchmarking script for AI Outlier Detection Pipeline.

This script measures:
1. Embedding generation speed
2. Outlier detection algorithm performance
3. Memory usage patterns
4. API response times
"""

import time
import psutil
import pandas as pd
import numpy as np
from typing import Dict, List
import json
import os
from contextlib import contextmanager

from src.pipeline import AIOutlierDetectionPipeline
from src.outlier_detectors import (
    EuclideanDetector, MahalanobisDetector, 
    LOFDetector, IsolationForestDetector
)

@contextmanager
def measure_performance():
    """Context manager to measure execution time and memory usage."""
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    execution_time = end_time - start_time
    memory_delta = end_memory - start_memory
    
    print(f"⏱️  Execution time: {execution_time:.2f}s")
    print(f"💾 Memory usage: {memory_delta:+.1f}MB (peak: {end_memory:.1f}MB)")

def generate_synthetic_data(n_samples: int = 1000) -> List[str]:
    """Generate synthetic text data for benchmarking."""
    templates = [
        "Machine learning algorithm {} processes data efficiently",
        "Deep neural network {} learns complex patterns",
        "Artificial intelligence system {} transforms industries",
        "Computer vision model {} interprets visual information",
        "Natural language processing {} understands human text",
        "Data science technique {} reveals hidden insights",
        "Statistical method {} analyzes numerical patterns",
        "Optimization algorithm {} finds optimal solutions"
    ]
    
    texts = []
    for i in range(n_samples):
        template = templates[i % len(templates)]
        text = template.format(f"v{i//len(templates) + 1}")
        texts.append(text)
    
    return texts

def benchmark_embedding_generation():
    """Benchmark embedding generation performance."""
    print("🔢 Benchmarking Embedding Generation")
    print("=" * 50)
    
    try:
        pipeline = AIOutlierDetectionPipeline()
        
        # Test different batch sizes
        batch_sizes = [10, 25, 50, 100]
        sample_sizes = [50, 100, 200, 500]
        
        results = []
        
        for n_samples in sample_sizes:
            texts = generate_synthetic_data(n_samples)
            
            for batch_size in batch_sizes:
                print(f"\n📊 Testing {n_samples} texts with batch size {batch_size}")
                
                # Update batch size
                pipeline.embedding_generator.batch_size = batch_size
                
                with measure_performance():
                    df = pd.DataFrame({
                        'Text': texts,
                        'Class Name': ['synthetic'] * len(texts),
                        'Label': range(len(texts))
                    })
                    
                    start_time = time.time()
                    df_with_embeddings = pipeline.embedding_generator.add_embeddings_to_dataframe(df)
                    end_time = time.time()
                
                throughput = n_samples / (end_time - start_time)
                
                results.append({
                    'n_samples': n_samples,
                    'batch_size': batch_size,
                    'execution_time': end_time - start_time,
                    'throughput': throughput
                })
                
                print(f"📈 Throughput: {throughput:.1f} docs/second")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv('benchmark_embedding_results.csv', index=False)
        print(f"\n💾 Results saved to benchmark_embedding_results.csv")
        
        # Show best configuration
        best_config = results_df.loc[results_df['throughput'].idxmax()]
        print(f"\n🏆 Best configuration:")
        print(f"   Batch size: {best_config['batch_size']}")
        print(f"   Throughput: {best_config['throughput']:.1f} docs/second")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

def benchmark_outlier_detectors():
    """Benchmark outlier detection algorithms."""
    print("\n🔍 Benchmarking Outlier Detection Algorithms")
    print("=" * 50)
    
    # Generate test data with known outliers
    np.random.seed(42)
    n_samples = 1000
    n_features = 384  # Typical embedding dimension
    
    # Create normal data
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=int(n_samples * 0.9)
    )
    
    # Create outliers
    outlier_data = np.random.uniform(-5, 5, (int(n_samples * 0.1), n_features))
    
    # Combine data
    all_data = np.vstack([normal_data, outlier_data])
    np.random.shuffle(all_data)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Text': [f'Document {i}' for i in range(len(all_data))],
        'Embeddings': [emb for emb in all_data],
        'Label': range(len(all_data))
    })
    
    # Test each detector
    detectors = [
        ('Euclidean', EuclideanDetector(radius=2.0)),
        ('Mahalanobis', MahalanobisDetector(confidence_level=0.95)),
        ('LOF', LOFDetector(n_neighbors=20)),
        ('IsolationForest', IsolationForestDetector(contamination=0.1))
    ]
    
    results = []
    
    for name, detector in detectors:
        print(f"\n🧪 Testing {name} detector")
        
        with measure_performance():
            start_time = time.time()
            result_df = detector.detect(df.copy())
            end_time = time.time()
        
        execution_time = end_time - start_time
        outlier_count = result_df[f'Outlier_{name}'].sum()
        throughput = len(df) / execution_time
        
        results.append({
            'detector': name,
            'execution_time': execution_time,
            'throughput': throughput,
            'outliers_detected': outlier_count,
            'outlier_percentage': (outlier_count / len(df)) * 100
        })
        
        print(f"📊 Outliers detected: {outlier_count} ({(outlier_count/len(df)*100):.1f}%)")
        print(f"📈 Throughput: {throughput:.1f} docs/second")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('benchmark_detector_results.csv', index=False)
    print(f"\n💾 Results saved to benchmark_detector_results.csv")
    
    # Show performance ranking
    print(f"\n🏆 Performance Ranking (by throughput):")
    sorted_results = results_df.sort_values('throughput', ascending=False)
    for i, row in sorted_results.iterrows():
        print(f"   {row['detector']}: {row['throughput']:.1f} docs/second")

def benchmark_full_pipeline():
    """Benchmark the complete pipeline."""
    print("\n🚀 Benchmarking Full Pipeline")
    print("=" * 50)
    
    try:
        pipeline = AIOutlierDetectionPipeline()
        
        print("🔄 Running complete pipeline benchmark...")
        
        with measure_performance():
            results = pipeline.run_full_pipeline(
                save_results=False,  # Skip saving for benchmark
                output_dir="benchmark_results"
            )
        
        print(f"\n📊 Pipeline Results:")
        print(f"   Documents processed: {results['total_documents']}")
        
        for method, count in results['summary_stats'].items():
            percentage = (count / results['total_documents']) * 100
            print(f"   {method}: {count} outliers ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"❌ Pipeline benchmark failed: {e}")

def benchmark_api_performance():
    """Benchmark API response times."""
    print("\n🌐 API Performance Benchmark")
    print("=" * 50)
    
    print("To benchmark API performance:")
    print("1. Start the API server: python api.py")
    print("2. Run load testing with tools like:")
    print("   - Apache Bench (ab)")
    print("   - wrk")
    print("   - Locust")
    
    print("\nExample commands:")
    print("# Test text analysis endpoint")
    print('ab -n 100 -c 10 -p test_payload.json -T application/json http://localhost:8000/analyze/texts')
    
    print("\n# Create test payload file:")
    test_payload = {
        "texts": ["Sample text for testing"] * 10,
        "categories": ["test"] * 10
    }
    
    with open('test_payload.json', 'w') as f:
        json.dump(test_payload, f)
    
    print("✅ Created test_payload.json for API testing")

def generate_benchmark_report():
    """Generate a comprehensive benchmark report."""
    print("\n📋 Generating Benchmark Report")
    print("=" * 50)
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': os.sys.version
        }
    }
    
    # Load benchmark results if they exist
    if os.path.exists('benchmark_embedding_results.csv'):
        embedding_df = pd.read_csv('benchmark_embedding_results.csv')
        report['embedding_performance'] = {
            'best_throughput': embedding_df['throughput'].max(),
            'best_batch_size': embedding_df.loc[embedding_df['throughput'].idxmax(), 'batch_size']
        }
    
    if os.path.exists('benchmark_detector_results.csv'):
        detector_df = pd.read_csv('benchmark_detector_results.csv')
        report['detector_performance'] = detector_df.to_dict('records')
    
    # Save report
    with open('benchmark_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("💾 Benchmark report saved to benchmark_report.json")

def main():
    """Run all benchmarks."""
    print("🏁 AI Outlier Detection Pipeline - Performance Benchmark")
    print("=" * 70)
    
    # Check system resources
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count()
    
    print(f"💻 System Info:")
    print(f"   CPU cores: {cpu_count}")
    print(f"   Memory: {memory_gb:.1f} GB")
    print(f"   Python: {os.sys.version.split()[0]}")
    
    if not os.getenv('NEBIUS_API_KEY'):
        print("\n⚠️  Warning: NEBIUS_API_KEY not found")
        print("Some benchmarks may fail without API configuration")
    
    try:
        benchmark_outlier_detectors()
        
        # Ask user about embedding benchmark (requires API)
        if os.getenv('NEBIUS_API_KEY'):
            response = input("\n🤔 Run embedding benchmark? This requires API calls (y/N): ")
            if response.lower() in ['y', 'yes']:
                benchmark_embedding_generation()
            
            response = input("\n🤔 Run full pipeline benchmark? This takes several minutes (y/N): ")
            if response.lower() in ['y', 'yes']:
                benchmark_full_pipeline()
        
        benchmark_api_performance()
        generate_benchmark_report()
        
        print("\n🎉 Benchmarking completed!")
        print("\nGenerated files:")
        print("- benchmark_detector_results.csv")
        print("- benchmark_embedding_results.csv (if run)")
        print("- benchmark_report.json")
        print("- test_payload.json")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")

if __name__ == "__main__":
    main()