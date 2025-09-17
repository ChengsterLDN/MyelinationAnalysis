import imagej
import scyjava
scyjava.config.add_options('-Xmx6g')

# initialize ImageJ

ij = imagej.init('sc.fiji:fiji:2.17.0')
print(f"ImageJ version: {ij.getVersion()}")