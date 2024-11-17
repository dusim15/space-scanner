try:
    from camera_interface import RoomMapper
    from advanced_perception import AdvancedPerceptionSystem
    from parallel_decision import ParallelDecisionSystem
    
    print("All imports successful!")
    
    # Test instantiation
    mapper = RoomMapper()
    print("RoomMapper instantiated successfully!")
    
except Exception as e:
    print(f"Import error: {e}") 