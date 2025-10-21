import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__)))
from src.evaluation.evaluation import run_comprehensive_evaluation
from src.evaluation.comparisons import run_all_comparisons
from src.visualization.visualizations import create_all_visualizations



def main():
    """
    Main function to run evaluation and create visualizations.
    """
    print("MNIST Classification Project")
    print("=" * 50)
    
    # 1. Run evaluation
    print("\nRunning evaluation...")
    
    evaluator = run_comprehensive_evaluation()
    results = evaluator.get_results_for_visualization()
    
    # 2. Run comparison analysis
    print("\nRunning comparison analysis...")
    comparison_analyzer = run_all_comparisons(results)
    
    # 3. Create visualizations
    print("\nCreating visualizations...")
    create_all_visualizations(results)
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProject interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("Please check your setup and try again.")
