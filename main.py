import dearpygui.dearpygui as dpg
from src.ServerControlWidget import ServerControlWidget
#import torchvision
#import torchvision.transforms as transforms
#from src.utils import dataset_split2
#
#transform           = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#local_training_data = torchvision.datasets.MNIST('./data', train=True , transform=transform, download=False)
#local_test_data     = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=False)
#
#a = dataset_split2(local_training_data, 10, "dirichlet", cncntrcn=0.01, plot=True)
                
dpg.create_context()
dpg.create_viewport(title="Federated learning with neural networks - MAG", width=1440, height=865)
dpg.setup_dearpygui()

sw1 = ServerControlWidget("ServerControl")
sw2 = ServerControlWidget("ServerControl2")
sw3 = ServerControlWidget("ServerControl3")

with dpg.theme() as window_style_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 5, category=dpg.mvThemeCat_Core)
        
dpg.bind_theme(window_style_theme)
dpg.show_viewport(maximized=False)
dpg.start_dearpygui()
#while dpg.is_dearpygui_running():
#
#    dpg.set_axis_limits(sw1.tag_x_axis_loss, dpg.get_axis_limits(sw1.tag_x_axis_acc)[0], dpg.get_axis_limits(sw1.tag_x_axis_acc)[1])
#    
#    dpg.render_dearpygui_frame()
dpg.destroy_context()