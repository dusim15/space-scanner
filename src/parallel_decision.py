from .advanced_perception import AdvancedPerceptionSystem
from obstacle_processor import ObstacleProcessor
from navigation_processor import NavigationProcessor
from hazard_processor import HazardProcessor
from interaction_processor import InteractionProcessor
import concurrent.futures

class ParallelDecisionSystem:
    def __init__(self, config_manager, event_system):
        self.config = config_manager
        self.events = event_system
        self.perception = AdvancedPerceptionSystem(config_manager, event_system)
        
        # Initialize decision processors
        self.processors = {
            'obstacle': ObstacleProcessor(),
            'navigation': NavigationProcessor(),
            'hazard': HazardProcessor(),
            'interaction': InteractionProcessor()
        }
        
    def process_environment(self, sensor_data):
        """Process environmental data in parallel"""
        try:
            # Get advanced perception results
            perception_results = self.perception.analyze_environment(sensor_data)
            
            # Process different aspects in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.processors)) as executor:
                futures = {
                    name: executor.submit(processor.process, perception_results)
                    for name, processor in self.processors.items()
                }
                
                # Gather results
                decisions = {
                    name: future.result()
                    for name, future in futures.items()
                }
                
            # Prioritize and combine decisions
            final_decision = self._prioritize_decisions(decisions)
            
            return final_decision
            
        except Exception as e:
            print(f"Decision processing error: {e}")
            return self._emergency_decision()
            
    def _prioritize_decisions(self, decisions):
        """Prioritize different decision aspects"""
        try:
            # Check for immediate hazards
            if decisions['hazard']['immediate_danger']:
                return self._emergency_decision()
                
            # Combine navigation and obstacle avoidance
            navigation_plan = self._combine_navigation_decisions(
                decisions['navigation'],
                decisions['obstacle']
            )
            
            # Adjust for material interactions
            final_plan = self._adjust_for_materials(
                navigation_plan,
                decisions['interaction']
            )
            
            return final_plan
            
        except Exception as e:
            print(f"Decision prioritization error: {e}")
            return self._emergency_decision() 

__all__ = ['ParallelDecisionSystem']