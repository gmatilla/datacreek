
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("Starting import diagnostics...")

try:
    print("Attempting to import datacreek.backend...")
    import datacreek.backend
    print("SUCCESS: Imported datacreek.backend")
except ImportError as e:
    print(f"FAILURE: Could not import datacreek.backend: {e}")
except Exception as e:
    print(f"FAILURE: Error importing datacreek.backend: {e}")

try:
    print("Attempting to import datacreek.core.knowledge_graph...")
    from datacreek.core.knowledge_graph import KnowledgeGraph
    print("SUCCESS: Imported KnowledgeGraph")
except ImportError as e:
    print(f"FAILURE: Could not import KnowledgeGraph: {e}")
except Exception as e:
    print(f"FAILURE: Error importing KnowledgeGraph: {e}")

try:
    print("Attempting to import datacreek.analysis.hybrid_ann...")
    import datacreek.analysis.hybrid_ann
    print("SUCCESS: Imported datacreek.analysis.hybrid_ann")
except ImportError as e:
    print(f"FAILURE: Could not import datacreek.analysis.hybrid_ann: {e}")
except Exception as e:
    print(f"FAILURE: Error importing datacreek.analysis.hybrid_ann: {e}")

try:
    print("Attempting to import datacreek.core.dataset_full...")
    import datacreek.core.dataset_full
    print("SUCCESS: Imported datacreek.core.dataset_full")
except ImportError as e:
    print(f"FAILURE: Could not import datacreek.core.dataset_full: {e}")
except Exception as e:
    print(f"FAILURE: Error importing datacreek.core.dataset_full: {e}")

print("Diagnostics complete.")
