from model import MyNet, ResNet18

mynet = MyNet()
resnet = ResNet18()

print("MyNet Architecture:")
print(mynet)
print("MyNet Parameters:", sum(p.numel() for p in mynet.parameters()))

print("\nResNet18 Architecture:")
print(resnet)
print("ResNet18 Parameters:", sum(p.numel() for p in resnet.parameters()))