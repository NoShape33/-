import sys
sys.path.insert(0, 'D:/Project_Mnist_Paper/src')

from generate_paper import build_relu_formula, build_softmax_formula, build_conv_formula
from lxml import etree

for name, func in [('ReLU', build_relu_formula), ('Softmax', build_softmax_formula), ('Conv', build_conv_formula)]:
    omath = func()
    print(f'\n{name}:')
    print(f'  tag: {omath.tag}')
    print(f'  children: {len(omath)}')
    s = etree.tostring(omath, encoding='unicode')
    print(f'  XML length: {len(s)}')
    print(f'  XML preview: {s[:150]}...')
