def test_all_imports():
    try:
        # Core components
        from camera_interface import RoomMapper
        from advanced_perception import AdvancedPerceptionSystem
        from parallel_decision import ParallelDecisionSystem
        from interaction_processor import InteractionProcessor
        
        print("✓ All core modules imported successfully!")
        
        # Test instantiation
        mapper = RoomMapper()
        perception = AdvancedPerceptionSystem()
        decision = ParallelDecisionSystem()
        interaction = InteractionProcessor()
        
        print("✓ All core classes instantiated successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("\n=== Testing Project Imports ===\n")
    success = test_all_imports()
    print("\n=== Test Complete ===")
    if not success:
        exit(1)  # Exit with error code if tests fail 