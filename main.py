from src.camera_interface import RoomMapper

def main():
    try:
        # Initialize RoomMapper
        mapper = RoomMapper()
        print("RoomMapper initialized successfully!")
        
        # Start your main application loop here
        while True:
            mapper.process_scan()
            # Add other necessary method calls
            mapper.update_display()
            mapper.handle_input()
            
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()