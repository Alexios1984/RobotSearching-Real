import rclpy
from rclpy.node import Node
# Importiamo i tuoi nuovi messaggi custom
from vmc_interfaces.msg import RobotConfiguration 
import sys

class ConfigSnapper(Node):
    def __init__(self):
        super().__init__('config_snapper')
        
        # Subscription to the topic for the initial configuration
        self.sub = self.create_subscription(
            RobotConfiguration, 
            '/vmc/initial_link_config', 
            self.listener_callback, 
            10
        )
        
        print("👀 Waiting for initial configuration message...")

    def listener_callback(self, msg):
        
        print("\n✅ The robot's initial configuration has been received:\n")
        print("="*60)
        
        # Iterate on the LinkData message to get each link's position
        for link_data in msg.links:
            name = link_data.link_name
            x = link_data.position.x
            y = link_data.position.y
            z = link_data.position.z
            
            print(f'\n    "{name}" : (x:{x:.4f}, y:{y:.4f}, z:{z:.4f}),')
            
        print("="*60)
        print("\n")
        
        # Destroy Node
        raise SystemExit

def main():
    rclpy.init()
    node = ConfigSnapper()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()