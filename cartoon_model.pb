
V
input_photoPlaceholder*
dtype0*-
shape$:"������������������
�
7generator/Conv/weights/Initializer/random_uniform/shapeConst*)
_class
loc:@generator/Conv/weights*
dtype0*%
valueB"             
�
5generator/Conv/weights/Initializer/random_uniform/minConst*)
_class
loc:@generator/Conv/weights*
dtype0*
valueB
 *�Er�
�
5generator/Conv/weights/Initializer/random_uniform/maxConst*)
_class
loc:@generator/Conv/weights*
dtype0*
valueB
 *�Er=
�
?generator/Conv/weights/Initializer/random_uniform/RandomUniformRandomUniform7generator/Conv/weights/Initializer/random_uniform/shape*
T0*)
_class
loc:@generator/Conv/weights*
dtype0*
seed2 *

seed 
�
5generator/Conv/weights/Initializer/random_uniform/subSub5generator/Conv/weights/Initializer/random_uniform/max5generator/Conv/weights/Initializer/random_uniform/min*
T0*)
_class
loc:@generator/Conv/weights
�
5generator/Conv/weights/Initializer/random_uniform/mulMul?generator/Conv/weights/Initializer/random_uniform/RandomUniform5generator/Conv/weights/Initializer/random_uniform/sub*
T0*)
_class
loc:@generator/Conv/weights
�
1generator/Conv/weights/Initializer/random_uniformAddV25generator/Conv/weights/Initializer/random_uniform/mul5generator/Conv/weights/Initializer/random_uniform/min*
T0*)
_class
loc:@generator/Conv/weights
�
generator/Conv/weightsVarHandleOp*)
_class
loc:@generator/Conv/weights*
allowed_devices
 *
	container *
dtype0*
shape: *'
shared_namegenerator/Conv/weights
e
7generator/Conv/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv/weights
�
generator/Conv/weights/AssignAssignVariableOpgenerator/Conv/weights1generator/Conv/weights/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
a
*generator/Conv/weights/Read/ReadVariableOpReadVariableOpgenerator/Conv/weights*
dtype0
�
'generator/Conv/biases/Initializer/zerosConst*(
_class
loc:@generator/Conv/biases*
dtype0*
valueB *    
�
generator/Conv/biasesVarHandleOp*(
_class
loc:@generator/Conv/biases*
allowed_devices
 *
	container *
dtype0*
shape: *&
shared_namegenerator/Conv/biases
c
6generator/Conv/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv/biases
�
generator/Conv/biases/AssignAssignVariableOpgenerator/Conv/biases'generator/Conv/biases/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
_
)generator/Conv/biases/Read/ReadVariableOpReadVariableOpgenerator/Conv/biases*
dtype0
[
$generator/Conv/Conv2D/ReadVariableOpReadVariableOpgenerator/Conv/weights*
dtype0
�
generator/Conv/Conv2DConv2Dinput_photo$generator/Conv/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
[
%generator/Conv/BiasAdd/ReadVariableOpReadVariableOpgenerator/Conv/biases*
dtype0

generator/Conv/BiasAddBiasAddgenerator/Conv/Conv2D%generator/Conv/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
Q
generator/LeakyRelu	LeakyRelugenerator/Conv/BiasAdd*
T0*
alpha%��L>
�
9generator/Conv_1/weights/Initializer/random_uniform/shapeConst*+
_class!
loc:@generator/Conv_1/weights*
dtype0*%
valueB"              
�
7generator/Conv_1/weights/Initializer/random_uniform/minConst*+
_class!
loc:@generator/Conv_1/weights*
dtype0*
valueB
 *�ѽ
�
7generator/Conv_1/weights/Initializer/random_uniform/maxConst*+
_class!
loc:@generator/Conv_1/weights*
dtype0*
valueB
 *��=
�
Agenerator/Conv_1/weights/Initializer/random_uniform/RandomUniformRandomUniform9generator/Conv_1/weights/Initializer/random_uniform/shape*
T0*+
_class!
loc:@generator/Conv_1/weights*
dtype0*
seed2 *

seed 
�
7generator/Conv_1/weights/Initializer/random_uniform/subSub7generator/Conv_1/weights/Initializer/random_uniform/max7generator/Conv_1/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@generator/Conv_1/weights
�
7generator/Conv_1/weights/Initializer/random_uniform/mulMulAgenerator/Conv_1/weights/Initializer/random_uniform/RandomUniform7generator/Conv_1/weights/Initializer/random_uniform/sub*
T0*+
_class!
loc:@generator/Conv_1/weights
�
3generator/Conv_1/weights/Initializer/random_uniformAddV27generator/Conv_1/weights/Initializer/random_uniform/mul7generator/Conv_1/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@generator/Conv_1/weights
�
generator/Conv_1/weightsVarHandleOp*+
_class!
loc:@generator/Conv_1/weights*
allowed_devices
 *
	container *
dtype0*
shape:  *)
shared_namegenerator/Conv_1/weights
i
9generator/Conv_1/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv_1/weights
�
generator/Conv_1/weights/AssignAssignVariableOpgenerator/Conv_1/weights3generator/Conv_1/weights/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
e
,generator/Conv_1/weights/Read/ReadVariableOpReadVariableOpgenerator/Conv_1/weights*
dtype0
�
)generator/Conv_1/biases/Initializer/zerosConst**
_class 
loc:@generator/Conv_1/biases*
dtype0*
valueB *    
�
generator/Conv_1/biasesVarHandleOp**
_class 
loc:@generator/Conv_1/biases*
allowed_devices
 *
	container *
dtype0*
shape: *(
shared_namegenerator/Conv_1/biases
g
8generator/Conv_1/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv_1/biases
�
generator/Conv_1/biases/AssignAssignVariableOpgenerator/Conv_1/biases)generator/Conv_1/biases/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
c
+generator/Conv_1/biases/Read/ReadVariableOpReadVariableOpgenerator/Conv_1/biases*
dtype0
_
&generator/Conv_1/Conv2D/ReadVariableOpReadVariableOpgenerator/Conv_1/weights*
dtype0
�
generator/Conv_1/Conv2DConv2Dgenerator/LeakyRelu&generator/Conv_1/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
_
'generator/Conv_1/BiasAdd/ReadVariableOpReadVariableOpgenerator/Conv_1/biases*
dtype0
�
generator/Conv_1/BiasAddBiasAddgenerator/Conv_1/Conv2D'generator/Conv_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
U
generator/LeakyRelu_1	LeakyRelugenerator/Conv_1/BiasAdd*
T0*
alpha%��L>
�
9generator/Conv_2/weights/Initializer/random_uniform/shapeConst*+
_class!
loc:@generator/Conv_2/weights*
dtype0*%
valueB"          @   
�
7generator/Conv_2/weights/Initializer/random_uniform/minConst*+
_class!
loc:@generator/Conv_2/weights*
dtype0*
valueB
 *����
�
7generator/Conv_2/weights/Initializer/random_uniform/maxConst*+
_class!
loc:@generator/Conv_2/weights*
dtype0*
valueB
 *���=
�
Agenerator/Conv_2/weights/Initializer/random_uniform/RandomUniformRandomUniform9generator/Conv_2/weights/Initializer/random_uniform/shape*
T0*+
_class!
loc:@generator/Conv_2/weights*
dtype0*
seed2 *

seed 
�
7generator/Conv_2/weights/Initializer/random_uniform/subSub7generator/Conv_2/weights/Initializer/random_uniform/max7generator/Conv_2/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@generator/Conv_2/weights
�
7generator/Conv_2/weights/Initializer/random_uniform/mulMulAgenerator/Conv_2/weights/Initializer/random_uniform/RandomUniform7generator/Conv_2/weights/Initializer/random_uniform/sub*
T0*+
_class!
loc:@generator/Conv_2/weights
�
3generator/Conv_2/weights/Initializer/random_uniformAddV27generator/Conv_2/weights/Initializer/random_uniform/mul7generator/Conv_2/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@generator/Conv_2/weights
�
generator/Conv_2/weightsVarHandleOp*+
_class!
loc:@generator/Conv_2/weights*
allowed_devices
 *
	container *
dtype0*
shape: @*)
shared_namegenerator/Conv_2/weights
i
9generator/Conv_2/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv_2/weights
�
generator/Conv_2/weights/AssignAssignVariableOpgenerator/Conv_2/weights3generator/Conv_2/weights/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
e
,generator/Conv_2/weights/Read/ReadVariableOpReadVariableOpgenerator/Conv_2/weights*
dtype0
�
)generator/Conv_2/biases/Initializer/zerosConst**
_class 
loc:@generator/Conv_2/biases*
dtype0*
valueB@*    
�
generator/Conv_2/biasesVarHandleOp**
_class 
loc:@generator/Conv_2/biases*
allowed_devices
 *
	container *
dtype0*
shape:@*(
shared_namegenerator/Conv_2/biases
g
8generator/Conv_2/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv_2/biases
�
generator/Conv_2/biases/AssignAssignVariableOpgenerator/Conv_2/biases)generator/Conv_2/biases/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
c
+generator/Conv_2/biases/Read/ReadVariableOpReadVariableOpgenerator/Conv_2/biases*
dtype0
_
&generator/Conv_2/Conv2D/ReadVariableOpReadVariableOpgenerator/Conv_2/weights*
dtype0
�
generator/Conv_2/Conv2DConv2Dgenerator/LeakyRelu_1&generator/Conv_2/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
_
'generator/Conv_2/BiasAdd/ReadVariableOpReadVariableOpgenerator/Conv_2/biases*
dtype0
�
generator/Conv_2/BiasAddBiasAddgenerator/Conv_2/Conv2D'generator/Conv_2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
U
generator/LeakyRelu_2	LeakyRelugenerator/Conv_2/BiasAdd*
T0*
alpha%��L>
�
9generator/Conv_3/weights/Initializer/random_uniform/shapeConst*+
_class!
loc:@generator/Conv_3/weights*
dtype0*%
valueB"      @   @   
�
7generator/Conv_3/weights/Initializer/random_uniform/minConst*+
_class!
loc:@generator/Conv_3/weights*
dtype0*
valueB
 *:͓�
�
7generator/Conv_3/weights/Initializer/random_uniform/maxConst*+
_class!
loc:@generator/Conv_3/weights*
dtype0*
valueB
 *:͓=
�
Agenerator/Conv_3/weights/Initializer/random_uniform/RandomUniformRandomUniform9generator/Conv_3/weights/Initializer/random_uniform/shape*
T0*+
_class!
loc:@generator/Conv_3/weights*
dtype0*
seed2 *

seed 
�
7generator/Conv_3/weights/Initializer/random_uniform/subSub7generator/Conv_3/weights/Initializer/random_uniform/max7generator/Conv_3/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@generator/Conv_3/weights
�
7generator/Conv_3/weights/Initializer/random_uniform/mulMulAgenerator/Conv_3/weights/Initializer/random_uniform/RandomUniform7generator/Conv_3/weights/Initializer/random_uniform/sub*
T0*+
_class!
loc:@generator/Conv_3/weights
�
3generator/Conv_3/weights/Initializer/random_uniformAddV27generator/Conv_3/weights/Initializer/random_uniform/mul7generator/Conv_3/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@generator/Conv_3/weights
�
generator/Conv_3/weightsVarHandleOp*+
_class!
loc:@generator/Conv_3/weights*
allowed_devices
 *
	container *
dtype0*
shape:@@*)
shared_namegenerator/Conv_3/weights
i
9generator/Conv_3/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv_3/weights
�
generator/Conv_3/weights/AssignAssignVariableOpgenerator/Conv_3/weights3generator/Conv_3/weights/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
e
,generator/Conv_3/weights/Read/ReadVariableOpReadVariableOpgenerator/Conv_3/weights*
dtype0
�
)generator/Conv_3/biases/Initializer/zerosConst**
_class 
loc:@generator/Conv_3/biases*
dtype0*
valueB@*    
�
generator/Conv_3/biasesVarHandleOp**
_class 
loc:@generator/Conv_3/biases*
allowed_devices
 *
	container *
dtype0*
shape:@*(
shared_namegenerator/Conv_3/biases
g
8generator/Conv_3/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv_3/biases
�
generator/Conv_3/biases/AssignAssignVariableOpgenerator/Conv_3/biases)generator/Conv_3/biases/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
c
+generator/Conv_3/biases/Read/ReadVariableOpReadVariableOpgenerator/Conv_3/biases*
dtype0
_
&generator/Conv_3/Conv2D/ReadVariableOpReadVariableOpgenerator/Conv_3/weights*
dtype0
�
generator/Conv_3/Conv2DConv2Dgenerator/LeakyRelu_2&generator/Conv_3/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
_
'generator/Conv_3/BiasAdd/ReadVariableOpReadVariableOpgenerator/Conv_3/biases*
dtype0
�
generator/Conv_3/BiasAddBiasAddgenerator/Conv_3/Conv2D'generator/Conv_3/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
U
generator/LeakyRelu_3	LeakyRelugenerator/Conv_3/BiasAdd*
T0*
alpha%��L>
�
9generator/Conv_4/weights/Initializer/random_uniform/shapeConst*+
_class!
loc:@generator/Conv_4/weights*
dtype0*%
valueB"      @   �   
�
7generator/Conv_4/weights/Initializer/random_uniform/minConst*+
_class!
loc:@generator/Conv_4/weights*
dtype0*
valueB
 *�[q�
�
7generator/Conv_4/weights/Initializer/random_uniform/maxConst*+
_class!
loc:@generator/Conv_4/weights*
dtype0*
valueB
 *�[q=
�
Agenerator/Conv_4/weights/Initializer/random_uniform/RandomUniformRandomUniform9generator/Conv_4/weights/Initializer/random_uniform/shape*
T0*+
_class!
loc:@generator/Conv_4/weights*
dtype0*
seed2 *

seed 
�
7generator/Conv_4/weights/Initializer/random_uniform/subSub7generator/Conv_4/weights/Initializer/random_uniform/max7generator/Conv_4/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@generator/Conv_4/weights
�
7generator/Conv_4/weights/Initializer/random_uniform/mulMulAgenerator/Conv_4/weights/Initializer/random_uniform/RandomUniform7generator/Conv_4/weights/Initializer/random_uniform/sub*
T0*+
_class!
loc:@generator/Conv_4/weights
�
3generator/Conv_4/weights/Initializer/random_uniformAddV27generator/Conv_4/weights/Initializer/random_uniform/mul7generator/Conv_4/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@generator/Conv_4/weights
�
generator/Conv_4/weightsVarHandleOp*+
_class!
loc:@generator/Conv_4/weights*
allowed_devices
 *
	container *
dtype0*
shape:@�*)
shared_namegenerator/Conv_4/weights
i
9generator/Conv_4/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv_4/weights
�
generator/Conv_4/weights/AssignAssignVariableOpgenerator/Conv_4/weights3generator/Conv_4/weights/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
e
,generator/Conv_4/weights/Read/ReadVariableOpReadVariableOpgenerator/Conv_4/weights*
dtype0
�
)generator/Conv_4/biases/Initializer/zerosConst**
_class 
loc:@generator/Conv_4/biases*
dtype0*
valueB�*    
�
generator/Conv_4/biasesVarHandleOp**
_class 
loc:@generator/Conv_4/biases*
allowed_devices
 *
	container *
dtype0*
shape:�*(
shared_namegenerator/Conv_4/biases
g
8generator/Conv_4/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv_4/biases
�
generator/Conv_4/biases/AssignAssignVariableOpgenerator/Conv_4/biases)generator/Conv_4/biases/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
c
+generator/Conv_4/biases/Read/ReadVariableOpReadVariableOpgenerator/Conv_4/biases*
dtype0
_
&generator/Conv_4/Conv2D/ReadVariableOpReadVariableOpgenerator/Conv_4/weights*
dtype0
�
generator/Conv_4/Conv2DConv2Dgenerator/LeakyRelu_3&generator/Conv_4/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
_
'generator/Conv_4/BiasAdd/ReadVariableOpReadVariableOpgenerator/Conv_4/biases*
dtype0
�
generator/Conv_4/BiasAddBiasAddgenerator/Conv_4/Conv2D'generator/Conv_4/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
U
generator/LeakyRelu_4	LeakyRelugenerator/Conv_4/BiasAdd*
T0*
alpha%��L>
�
@generator/block_0/conv1/weights/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@generator/block_0/conv1/weights*
dtype0*%
valueB"      �   �   
�
>generator/block_0/conv1/weights/Initializer/random_uniform/minConst*2
_class(
&$loc:@generator/block_0/conv1/weights*
dtype0*
valueB
 *�Q�
�
>generator/block_0/conv1/weights/Initializer/random_uniform/maxConst*2
_class(
&$loc:@generator/block_0/conv1/weights*
dtype0*
valueB
 *�Q=
�
Hgenerator/block_0/conv1/weights/Initializer/random_uniform/RandomUniformRandomUniform@generator/block_0/conv1/weights/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@generator/block_0/conv1/weights*
dtype0*
seed2 *

seed 
�
>generator/block_0/conv1/weights/Initializer/random_uniform/subSub>generator/block_0/conv1/weights/Initializer/random_uniform/max>generator/block_0/conv1/weights/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@generator/block_0/conv1/weights
�
>generator/block_0/conv1/weights/Initializer/random_uniform/mulMulHgenerator/block_0/conv1/weights/Initializer/random_uniform/RandomUniform>generator/block_0/conv1/weights/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@generator/block_0/conv1/weights
�
:generator/block_0/conv1/weights/Initializer/random_uniformAddV2>generator/block_0/conv1/weights/Initializer/random_uniform/mul>generator/block_0/conv1/weights/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@generator/block_0/conv1/weights
�
generator/block_0/conv1/weightsVarHandleOp*2
_class(
&$loc:@generator/block_0/conv1/weights*
allowed_devices
 *
	container *
dtype0*
shape:��*0
shared_name!generator/block_0/conv1/weights
w
@generator/block_0/conv1/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/block_0/conv1/weights
�
&generator/block_0/conv1/weights/AssignAssignVariableOpgenerator/block_0/conv1/weights:generator/block_0/conv1/weights/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
s
3generator/block_0/conv1/weights/Read/ReadVariableOpReadVariableOpgenerator/block_0/conv1/weights*
dtype0
�
0generator/block_0/conv1/biases/Initializer/zerosConst*1
_class'
%#loc:@generator/block_0/conv1/biases*
dtype0*
valueB�*    
�
generator/block_0/conv1/biasesVarHandleOp*1
_class'
%#loc:@generator/block_0/conv1/biases*
allowed_devices
 *
	container *
dtype0*
shape:�*/
shared_name generator/block_0/conv1/biases
u
?generator/block_0/conv1/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/block_0/conv1/biases
�
%generator/block_0/conv1/biases/AssignAssignVariableOpgenerator/block_0/conv1/biases0generator/block_0/conv1/biases/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
q
2generator/block_0/conv1/biases/Read/ReadVariableOpReadVariableOpgenerator/block_0/conv1/biases*
dtype0
m
-generator/block_0/conv1/Conv2D/ReadVariableOpReadVariableOpgenerator/block_0/conv1/weights*
dtype0
�
generator/block_0/conv1/Conv2DConv2Dgenerator/LeakyRelu_4-generator/block_0/conv1/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
m
.generator/block_0/conv1/BiasAdd/ReadVariableOpReadVariableOpgenerator/block_0/conv1/biases*
dtype0
�
generator/block_0/conv1/BiasAddBiasAddgenerator/block_0/conv1/Conv2D.generator/block_0/conv1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
b
generator/block_0/LeakyRelu	LeakyRelugenerator/block_0/conv1/BiasAdd*
T0*
alpha%��L>
�
@generator/block_0/conv2/weights/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@generator/block_0/conv2/weights*
dtype0*%
valueB"      �   �   
�
>generator/block_0/conv2/weights/Initializer/random_uniform/minConst*2
_class(
&$loc:@generator/block_0/conv2/weights*
dtype0*
valueB
 *�Q�
�
>generator/block_0/conv2/weights/Initializer/random_uniform/maxConst*2
_class(
&$loc:@generator/block_0/conv2/weights*
dtype0*
valueB
 *�Q=
�
Hgenerator/block_0/conv2/weights/Initializer/random_uniform/RandomUniformRandomUniform@generator/block_0/conv2/weights/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@generator/block_0/conv2/weights*
dtype0*
seed2 *

seed 
�
>generator/block_0/conv2/weights/Initializer/random_uniform/subSub>generator/block_0/conv2/weights/Initializer/random_uniform/max>generator/block_0/conv2/weights/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@generator/block_0/conv2/weights
�
>generator/block_0/conv2/weights/Initializer/random_uniform/mulMulHgenerator/block_0/conv2/weights/Initializer/random_uniform/RandomUniform>generator/block_0/conv2/weights/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@generator/block_0/conv2/weights
�
:generator/block_0/conv2/weights/Initializer/random_uniformAddV2>generator/block_0/conv2/weights/Initializer/random_uniform/mul>generator/block_0/conv2/weights/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@generator/block_0/conv2/weights
�
generator/block_0/conv2/weightsVarHandleOp*2
_class(
&$loc:@generator/block_0/conv2/weights*
allowed_devices
 *
	container *
dtype0*
shape:��*0
shared_name!generator/block_0/conv2/weights
w
@generator/block_0/conv2/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/block_0/conv2/weights
�
&generator/block_0/conv2/weights/AssignAssignVariableOpgenerator/block_0/conv2/weights:generator/block_0/conv2/weights/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
s
3generator/block_0/conv2/weights/Read/ReadVariableOpReadVariableOpgenerator/block_0/conv2/weights*
dtype0
�
0generator/block_0/conv2/biases/Initializer/zerosConst*1
_class'
%#loc:@generator/block_0/conv2/biases*
dtype0*
valueB�*    
�
generator/block_0/conv2/biasesVarHandleOp*1
_class'
%#loc:@generator/block_0/conv2/biases*
allowed_devices
 *
	container *
dtype0*
shape:�*/
shared_name generator/block_0/conv2/biases
u
?generator/block_0/conv2/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/block_0/conv2/biases
�
%generator/block_0/conv2/biases/AssignAssignVariableOpgenerator/block_0/conv2/biases0generator/block_0/conv2/biases/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
q
2generator/block_0/conv2/biases/Read/ReadVariableOpReadVariableOpgenerator/block_0/conv2/biases*
dtype0
m
-generator/block_0/conv2/Conv2D/ReadVariableOpReadVariableOpgenerator/block_0/conv2/weights*
dtype0
�
generator/block_0/conv2/Conv2DConv2Dgenerator/block_0/LeakyRelu-generator/block_0/conv2/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
m
.generator/block_0/conv2/BiasAdd/ReadVariableOpReadVariableOpgenerator/block_0/conv2/biases*
dtype0
�
generator/block_0/conv2/BiasAddBiasAddgenerator/block_0/conv2/Conv2D.generator/block_0/conv2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
_
generator/block_0/addAddV2generator/block_0/conv2/BiasAddgenerator/LeakyRelu_4*
T0
�
@generator/block_1/conv1/weights/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@generator/block_1/conv1/weights*
dtype0*%
valueB"      �   �   
�
>generator/block_1/conv1/weights/Initializer/random_uniform/minConst*2
_class(
&$loc:@generator/block_1/conv1/weights*
dtype0*
valueB
 *�Q�
�
>generator/block_1/conv1/weights/Initializer/random_uniform/maxConst*2
_class(
&$loc:@generator/block_1/conv1/weights*
dtype0*
valueB
 *�Q=
�
Hgenerator/block_1/conv1/weights/Initializer/random_uniform/RandomUniformRandomUniform@generator/block_1/conv1/weights/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@generator/block_1/conv1/weights*
dtype0*
seed2 *

seed 
�
>generator/block_1/conv1/weights/Initializer/random_uniform/subSub>generator/block_1/conv1/weights/Initializer/random_uniform/max>generator/block_1/conv1/weights/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@generator/block_1/conv1/weights
�
>generator/block_1/conv1/weights/Initializer/random_uniform/mulMulHgenerator/block_1/conv1/weights/Initializer/random_uniform/RandomUniform>generator/block_1/conv1/weights/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@generator/block_1/conv1/weights
�
:generator/block_1/conv1/weights/Initializer/random_uniformAddV2>generator/block_1/conv1/weights/Initializer/random_uniform/mul>generator/block_1/conv1/weights/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@generator/block_1/conv1/weights
�
generator/block_1/conv1/weightsVarHandleOp*2
_class(
&$loc:@generator/block_1/conv1/weights*
allowed_devices
 *
	container *
dtype0*
shape:��*0
shared_name!generator/block_1/conv1/weights
w
@generator/block_1/conv1/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/block_1/conv1/weights
�
&generator/block_1/conv1/weights/AssignAssignVariableOpgenerator/block_1/conv1/weights:generator/block_1/conv1/weights/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
s
3generator/block_1/conv1/weights/Read/ReadVariableOpReadVariableOpgenerator/block_1/conv1/weights*
dtype0
�
0generator/block_1/conv1/biases/Initializer/zerosConst*1
_class'
%#loc:@generator/block_1/conv1/biases*
dtype0*
valueB�*    
�
generator/block_1/conv1/biasesVarHandleOp*1
_class'
%#loc:@generator/block_1/conv1/biases*
allowed_devices
 *
	container *
dtype0*
shape:�*/
shared_name generator/block_1/conv1/biases
u
?generator/block_1/conv1/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/block_1/conv1/biases
�
%generator/block_1/conv1/biases/AssignAssignVariableOpgenerator/block_1/conv1/biases0generator/block_1/conv1/biases/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
q
2generator/block_1/conv1/biases/Read/ReadVariableOpReadVariableOpgenerator/block_1/conv1/biases*
dtype0
m
-generator/block_1/conv1/Conv2D/ReadVariableOpReadVariableOpgenerator/block_1/conv1/weights*
dtype0
�
generator/block_1/conv1/Conv2DConv2Dgenerator/block_0/add-generator/block_1/conv1/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
m
.generator/block_1/conv1/BiasAdd/ReadVariableOpReadVariableOpgenerator/block_1/conv1/biases*
dtype0
�
generator/block_1/conv1/BiasAddBiasAddgenerator/block_1/conv1/Conv2D.generator/block_1/conv1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
b
generator/block_1/LeakyRelu	LeakyRelugenerator/block_1/conv1/BiasAdd*
T0*
alpha%��L>
�
@generator/block_1/conv2/weights/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@generator/block_1/conv2/weights*
dtype0*%
valueB"      �   �   
�
>generator/block_1/conv2/weights/Initializer/random_uniform/minConst*2
_class(
&$loc:@generator/block_1/conv2/weights*
dtype0*
valueB
 *�Q�
�
>generator/block_1/conv2/weights/Initializer/random_uniform/maxConst*2
_class(
&$loc:@generator/block_1/conv2/weights*
dtype0*
valueB
 *�Q=
�
Hgenerator/block_1/conv2/weights/Initializer/random_uniform/RandomUniformRandomUniform@generator/block_1/conv2/weights/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@generator/block_1/conv2/weights*
dtype0*
seed2 *

seed 
�
>generator/block_1/conv2/weights/Initializer/random_uniform/subSub>generator/block_1/conv2/weights/Initializer/random_uniform/max>generator/block_1/conv2/weights/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@generator/block_1/conv2/weights
�
>generator/block_1/conv2/weights/Initializer/random_uniform/mulMulHgenerator/block_1/conv2/weights/Initializer/random_uniform/RandomUniform>generator/block_1/conv2/weights/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@generator/block_1/conv2/weights
�
:generator/block_1/conv2/weights/Initializer/random_uniformAddV2>generator/block_1/conv2/weights/Initializer/random_uniform/mul>generator/block_1/conv2/weights/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@generator/block_1/conv2/weights
�
generator/block_1/conv2/weightsVarHandleOp*2
_class(
&$loc:@generator/block_1/conv2/weights*
allowed_devices
 *
	container *
dtype0*
shape:��*0
shared_name!generator/block_1/conv2/weights
w
@generator/block_1/conv2/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/block_1/conv2/weights
�
&generator/block_1/conv2/weights/AssignAssignVariableOpgenerator/block_1/conv2/weights:generator/block_1/conv2/weights/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
s
3generator/block_1/conv2/weights/Read/ReadVariableOpReadVariableOpgenerator/block_1/conv2/weights*
dtype0
�
0generator/block_1/conv2/biases/Initializer/zerosConst*1
_class'
%#loc:@generator/block_1/conv2/biases*
dtype0*
valueB�*    
�
generator/block_1/conv2/biasesVarHandleOp*1
_class'
%#loc:@generator/block_1/conv2/biases*
allowed_devices
 *
	container *
dtype0*
shape:�*/
shared_name generator/block_1/conv2/biases
u
?generator/block_1/conv2/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/block_1/conv2/biases
�
%generator/block_1/conv2/biases/AssignAssignVariableOpgenerator/block_1/conv2/biases0generator/block_1/conv2/biases/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
q
2generator/block_1/conv2/biases/Read/ReadVariableOpReadVariableOpgenerator/block_1/conv2/biases*
dtype0
m
-generator/block_1/conv2/Conv2D/ReadVariableOpReadVariableOpgenerator/block_1/conv2/weights*
dtype0
�
generator/block_1/conv2/Conv2DConv2Dgenerator/block_1/LeakyRelu-generator/block_1/conv2/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
m
.generator/block_1/conv2/BiasAdd/ReadVariableOpReadVariableOpgenerator/block_1/conv2/biases*
dtype0
�
generator/block_1/conv2/BiasAddBiasAddgenerator/block_1/conv2/Conv2D.generator/block_1/conv2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
_
generator/block_1/addAddV2generator/block_1/conv2/BiasAddgenerator/block_0/add*
T0
�
@generator/block_2/conv1/weights/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@generator/block_2/conv1/weights*
dtype0*%
valueB"      �   �   
�
>generator/block_2/conv1/weights/Initializer/random_uniform/minConst*2
_class(
&$loc:@generator/block_2/conv1/weights*
dtype0*
valueB
 *�Q�
�
>generator/block_2/conv1/weights/Initializer/random_uniform/maxConst*2
_class(
&$loc:@generator/block_2/conv1/weights*
dtype0*
valueB
 *�Q=
�
Hgenerator/block_2/conv1/weights/Initializer/random_uniform/RandomUniformRandomUniform@generator/block_2/conv1/weights/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@generator/block_2/conv1/weights*
dtype0*
seed2 *

seed 
�
>generator/block_2/conv1/weights/Initializer/random_uniform/subSub>generator/block_2/conv1/weights/Initializer/random_uniform/max>generator/block_2/conv1/weights/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@generator/block_2/conv1/weights
�
>generator/block_2/conv1/weights/Initializer/random_uniform/mulMulHgenerator/block_2/conv1/weights/Initializer/random_uniform/RandomUniform>generator/block_2/conv1/weights/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@generator/block_2/conv1/weights
�
:generator/block_2/conv1/weights/Initializer/random_uniformAddV2>generator/block_2/conv1/weights/Initializer/random_uniform/mul>generator/block_2/conv1/weights/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@generator/block_2/conv1/weights
�
generator/block_2/conv1/weightsVarHandleOp*2
_class(
&$loc:@generator/block_2/conv1/weights*
allowed_devices
 *
	container *
dtype0*
shape:��*0
shared_name!generator/block_2/conv1/weights
w
@generator/block_2/conv1/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/block_2/conv1/weights
�
&generator/block_2/conv1/weights/AssignAssignVariableOpgenerator/block_2/conv1/weights:generator/block_2/conv1/weights/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
s
3generator/block_2/conv1/weights/Read/ReadVariableOpReadVariableOpgenerator/block_2/conv1/weights*
dtype0
�
0generator/block_2/conv1/biases/Initializer/zerosConst*1
_class'
%#loc:@generator/block_2/conv1/biases*
dtype0*
valueB�*    
�
generator/block_2/conv1/biasesVarHandleOp*1
_class'
%#loc:@generator/block_2/conv1/biases*
allowed_devices
 *
	container *
dtype0*
shape:�*/
shared_name generator/block_2/conv1/biases
u
?generator/block_2/conv1/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/block_2/conv1/biases
�
%generator/block_2/conv1/biases/AssignAssignVariableOpgenerator/block_2/conv1/biases0generator/block_2/conv1/biases/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
q
2generator/block_2/conv1/biases/Read/ReadVariableOpReadVariableOpgenerator/block_2/conv1/biases*
dtype0
m
-generator/block_2/conv1/Conv2D/ReadVariableOpReadVariableOpgenerator/block_2/conv1/weights*
dtype0
�
generator/block_2/conv1/Conv2DConv2Dgenerator/block_1/add-generator/block_2/conv1/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
m
.generator/block_2/conv1/BiasAdd/ReadVariableOpReadVariableOpgenerator/block_2/conv1/biases*
dtype0
�
generator/block_2/conv1/BiasAddBiasAddgenerator/block_2/conv1/Conv2D.generator/block_2/conv1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
b
generator/block_2/LeakyRelu	LeakyRelugenerator/block_2/conv1/BiasAdd*
T0*
alpha%��L>
�
@generator/block_2/conv2/weights/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@generator/block_2/conv2/weights*
dtype0*%
valueB"      �   �   
�
>generator/block_2/conv2/weights/Initializer/random_uniform/minConst*2
_class(
&$loc:@generator/block_2/conv2/weights*
dtype0*
valueB
 *�Q�
�
>generator/block_2/conv2/weights/Initializer/random_uniform/maxConst*2
_class(
&$loc:@generator/block_2/conv2/weights*
dtype0*
valueB
 *�Q=
�
Hgenerator/block_2/conv2/weights/Initializer/random_uniform/RandomUniformRandomUniform@generator/block_2/conv2/weights/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@generator/block_2/conv2/weights*
dtype0*
seed2 *

seed 
�
>generator/block_2/conv2/weights/Initializer/random_uniform/subSub>generator/block_2/conv2/weights/Initializer/random_uniform/max>generator/block_2/conv2/weights/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@generator/block_2/conv2/weights
�
>generator/block_2/conv2/weights/Initializer/random_uniform/mulMulHgenerator/block_2/conv2/weights/Initializer/random_uniform/RandomUniform>generator/block_2/conv2/weights/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@generator/block_2/conv2/weights
�
:generator/block_2/conv2/weights/Initializer/random_uniformAddV2>generator/block_2/conv2/weights/Initializer/random_uniform/mul>generator/block_2/conv2/weights/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@generator/block_2/conv2/weights
�
generator/block_2/conv2/weightsVarHandleOp*2
_class(
&$loc:@generator/block_2/conv2/weights*
allowed_devices
 *
	container *
dtype0*
shape:��*0
shared_name!generator/block_2/conv2/weights
w
@generator/block_2/conv2/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/block_2/conv2/weights
�
&generator/block_2/conv2/weights/AssignAssignVariableOpgenerator/block_2/conv2/weights:generator/block_2/conv2/weights/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
s
3generator/block_2/conv2/weights/Read/ReadVariableOpReadVariableOpgenerator/block_2/conv2/weights*
dtype0
�
0generator/block_2/conv2/biases/Initializer/zerosConst*1
_class'
%#loc:@generator/block_2/conv2/biases*
dtype0*
valueB�*    
�
generator/block_2/conv2/biasesVarHandleOp*1
_class'
%#loc:@generator/block_2/conv2/biases*
allowed_devices
 *
	container *
dtype0*
shape:�*/
shared_name generator/block_2/conv2/biases
u
?generator/block_2/conv2/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/block_2/conv2/biases
�
%generator/block_2/conv2/biases/AssignAssignVariableOpgenerator/block_2/conv2/biases0generator/block_2/conv2/biases/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
q
2generator/block_2/conv2/biases/Read/ReadVariableOpReadVariableOpgenerator/block_2/conv2/biases*
dtype0
m
-generator/block_2/conv2/Conv2D/ReadVariableOpReadVariableOpgenerator/block_2/conv2/weights*
dtype0
�
generator/block_2/conv2/Conv2DConv2Dgenerator/block_2/LeakyRelu-generator/block_2/conv2/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
m
.generator/block_2/conv2/BiasAdd/ReadVariableOpReadVariableOpgenerator/block_2/conv2/biases*
dtype0
�
generator/block_2/conv2/BiasAddBiasAddgenerator/block_2/conv2/Conv2D.generator/block_2/conv2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
_
generator/block_2/addAddV2generator/block_2/conv2/BiasAddgenerator/block_1/add*
T0
�
@generator/block_3/conv1/weights/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@generator/block_3/conv1/weights*
dtype0*%
valueB"      �   �   
�
>generator/block_3/conv1/weights/Initializer/random_uniform/minConst*2
_class(
&$loc:@generator/block_3/conv1/weights*
dtype0*
valueB
 *�Q�
�
>generator/block_3/conv1/weights/Initializer/random_uniform/maxConst*2
_class(
&$loc:@generator/block_3/conv1/weights*
dtype0*
valueB
 *�Q=
�
Hgenerator/block_3/conv1/weights/Initializer/random_uniform/RandomUniformRandomUniform@generator/block_3/conv1/weights/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@generator/block_3/conv1/weights*
dtype0*
seed2 *

seed 
�
>generator/block_3/conv1/weights/Initializer/random_uniform/subSub>generator/block_3/conv1/weights/Initializer/random_uniform/max>generator/block_3/conv1/weights/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@generator/block_3/conv1/weights
�
>generator/block_3/conv1/weights/Initializer/random_uniform/mulMulHgenerator/block_3/conv1/weights/Initializer/random_uniform/RandomUniform>generator/block_3/conv1/weights/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@generator/block_3/conv1/weights
�
:generator/block_3/conv1/weights/Initializer/random_uniformAddV2>generator/block_3/conv1/weights/Initializer/random_uniform/mul>generator/block_3/conv1/weights/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@generator/block_3/conv1/weights
�
generator/block_3/conv1/weightsVarHandleOp*2
_class(
&$loc:@generator/block_3/conv1/weights*
allowed_devices
 *
	container *
dtype0*
shape:��*0
shared_name!generator/block_3/conv1/weights
w
@generator/block_3/conv1/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/block_3/conv1/weights
�
&generator/block_3/conv1/weights/AssignAssignVariableOpgenerator/block_3/conv1/weights:generator/block_3/conv1/weights/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
s
3generator/block_3/conv1/weights/Read/ReadVariableOpReadVariableOpgenerator/block_3/conv1/weights*
dtype0
�
0generator/block_3/conv1/biases/Initializer/zerosConst*1
_class'
%#loc:@generator/block_3/conv1/biases*
dtype0*
valueB�*    
�
generator/block_3/conv1/biasesVarHandleOp*1
_class'
%#loc:@generator/block_3/conv1/biases*
allowed_devices
 *
	container *
dtype0*
shape:�*/
shared_name generator/block_3/conv1/biases
u
?generator/block_3/conv1/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/block_3/conv1/biases
�
%generator/block_3/conv1/biases/AssignAssignVariableOpgenerator/block_3/conv1/biases0generator/block_3/conv1/biases/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
q
2generator/block_3/conv1/biases/Read/ReadVariableOpReadVariableOpgenerator/block_3/conv1/biases*
dtype0
m
-generator/block_3/conv1/Conv2D/ReadVariableOpReadVariableOpgenerator/block_3/conv1/weights*
dtype0
�
generator/block_3/conv1/Conv2DConv2Dgenerator/block_2/add-generator/block_3/conv1/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
m
.generator/block_3/conv1/BiasAdd/ReadVariableOpReadVariableOpgenerator/block_3/conv1/biases*
dtype0
�
generator/block_3/conv1/BiasAddBiasAddgenerator/block_3/conv1/Conv2D.generator/block_3/conv1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
b
generator/block_3/LeakyRelu	LeakyRelugenerator/block_3/conv1/BiasAdd*
T0*
alpha%��L>
�
@generator/block_3/conv2/weights/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@generator/block_3/conv2/weights*
dtype0*%
valueB"      �   �   
�
>generator/block_3/conv2/weights/Initializer/random_uniform/minConst*2
_class(
&$loc:@generator/block_3/conv2/weights*
dtype0*
valueB
 *�Q�
�
>generator/block_3/conv2/weights/Initializer/random_uniform/maxConst*2
_class(
&$loc:@generator/block_3/conv2/weights*
dtype0*
valueB
 *�Q=
�
Hgenerator/block_3/conv2/weights/Initializer/random_uniform/RandomUniformRandomUniform@generator/block_3/conv2/weights/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@generator/block_3/conv2/weights*
dtype0*
seed2 *

seed 
�
>generator/block_3/conv2/weights/Initializer/random_uniform/subSub>generator/block_3/conv2/weights/Initializer/random_uniform/max>generator/block_3/conv2/weights/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@generator/block_3/conv2/weights
�
>generator/block_3/conv2/weights/Initializer/random_uniform/mulMulHgenerator/block_3/conv2/weights/Initializer/random_uniform/RandomUniform>generator/block_3/conv2/weights/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@generator/block_3/conv2/weights
�
:generator/block_3/conv2/weights/Initializer/random_uniformAddV2>generator/block_3/conv2/weights/Initializer/random_uniform/mul>generator/block_3/conv2/weights/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@generator/block_3/conv2/weights
�
generator/block_3/conv2/weightsVarHandleOp*2
_class(
&$loc:@generator/block_3/conv2/weights*
allowed_devices
 *
	container *
dtype0*
shape:��*0
shared_name!generator/block_3/conv2/weights
w
@generator/block_3/conv2/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/block_3/conv2/weights
�
&generator/block_3/conv2/weights/AssignAssignVariableOpgenerator/block_3/conv2/weights:generator/block_3/conv2/weights/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
s
3generator/block_3/conv2/weights/Read/ReadVariableOpReadVariableOpgenerator/block_3/conv2/weights*
dtype0
�
0generator/block_3/conv2/biases/Initializer/zerosConst*1
_class'
%#loc:@generator/block_3/conv2/biases*
dtype0*
valueB�*    
�
generator/block_3/conv2/biasesVarHandleOp*1
_class'
%#loc:@generator/block_3/conv2/biases*
allowed_devices
 *
	container *
dtype0*
shape:�*/
shared_name generator/block_3/conv2/biases
u
?generator/block_3/conv2/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/block_3/conv2/biases
�
%generator/block_3/conv2/biases/AssignAssignVariableOpgenerator/block_3/conv2/biases0generator/block_3/conv2/biases/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
q
2generator/block_3/conv2/biases/Read/ReadVariableOpReadVariableOpgenerator/block_3/conv2/biases*
dtype0
m
-generator/block_3/conv2/Conv2D/ReadVariableOpReadVariableOpgenerator/block_3/conv2/weights*
dtype0
�
generator/block_3/conv2/Conv2DConv2Dgenerator/block_3/LeakyRelu-generator/block_3/conv2/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
m
.generator/block_3/conv2/BiasAdd/ReadVariableOpReadVariableOpgenerator/block_3/conv2/biases*
dtype0
�
generator/block_3/conv2/BiasAddBiasAddgenerator/block_3/conv2/Conv2D.generator/block_3/conv2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
_
generator/block_3/addAddV2generator/block_3/conv2/BiasAddgenerator/block_2/add*
T0
�
9generator/Conv_5/weights/Initializer/random_uniform/shapeConst*+
_class!
loc:@generator/Conv_5/weights*
dtype0*%
valueB"      �   @   
�
7generator/Conv_5/weights/Initializer/random_uniform/minConst*+
_class!
loc:@generator/Conv_5/weights*
dtype0*
valueB
 *�[q�
�
7generator/Conv_5/weights/Initializer/random_uniform/maxConst*+
_class!
loc:@generator/Conv_5/weights*
dtype0*
valueB
 *�[q=
�
Agenerator/Conv_5/weights/Initializer/random_uniform/RandomUniformRandomUniform9generator/Conv_5/weights/Initializer/random_uniform/shape*
T0*+
_class!
loc:@generator/Conv_5/weights*
dtype0*
seed2 *

seed 
�
7generator/Conv_5/weights/Initializer/random_uniform/subSub7generator/Conv_5/weights/Initializer/random_uniform/max7generator/Conv_5/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@generator/Conv_5/weights
�
7generator/Conv_5/weights/Initializer/random_uniform/mulMulAgenerator/Conv_5/weights/Initializer/random_uniform/RandomUniform7generator/Conv_5/weights/Initializer/random_uniform/sub*
T0*+
_class!
loc:@generator/Conv_5/weights
�
3generator/Conv_5/weights/Initializer/random_uniformAddV27generator/Conv_5/weights/Initializer/random_uniform/mul7generator/Conv_5/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@generator/Conv_5/weights
�
generator/Conv_5/weightsVarHandleOp*+
_class!
loc:@generator/Conv_5/weights*
allowed_devices
 *
	container *
dtype0*
shape:�@*)
shared_namegenerator/Conv_5/weights
i
9generator/Conv_5/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv_5/weights
�
generator/Conv_5/weights/AssignAssignVariableOpgenerator/Conv_5/weights3generator/Conv_5/weights/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
e
,generator/Conv_5/weights/Read/ReadVariableOpReadVariableOpgenerator/Conv_5/weights*
dtype0
�
)generator/Conv_5/biases/Initializer/zerosConst**
_class 
loc:@generator/Conv_5/biases*
dtype0*
valueB@*    
�
generator/Conv_5/biasesVarHandleOp**
_class 
loc:@generator/Conv_5/biases*
allowed_devices
 *
	container *
dtype0*
shape:@*(
shared_namegenerator/Conv_5/biases
g
8generator/Conv_5/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv_5/biases
�
generator/Conv_5/biases/AssignAssignVariableOpgenerator/Conv_5/biases)generator/Conv_5/biases/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
c
+generator/Conv_5/biases/Read/ReadVariableOpReadVariableOpgenerator/Conv_5/biases*
dtype0
_
&generator/Conv_5/Conv2D/ReadVariableOpReadVariableOpgenerator/Conv_5/weights*
dtype0
�
generator/Conv_5/Conv2DConv2Dgenerator/block_3/add&generator/Conv_5/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
_
'generator/Conv_5/BiasAdd/ReadVariableOpReadVariableOpgenerator/Conv_5/biases*
dtype0
�
generator/Conv_5/BiasAddBiasAddgenerator/Conv_5/Conv2D'generator/Conv_5/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
U
generator/LeakyRelu_5	LeakyRelugenerator/Conv_5/BiasAdd*
T0*
alpha%��L>
V
generator/ShapeShapegenerator/LeakyRelu_5*
T0*
out_type0:��
K
generator/strided_slice/stackConst*
dtype0*
valueB:
M
generator/strided_slice/stack_1Const*
dtype0*
valueB:
M
generator/strided_slice/stack_2Const*
dtype0*
valueB:
�
generator/strided_sliceStridedSlicegenerator/Shapegenerator/strided_slice/stackgenerator/strided_slice/stack_1generator/strided_slice/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
X
generator/Shape_1Shapegenerator/LeakyRelu_5*
T0*
out_type0:��
M
generator/strided_slice_1/stackConst*
dtype0*
valueB:
O
!generator/strided_slice_1/stack_1Const*
dtype0*
valueB:
O
!generator/strided_slice_1/stack_2Const*
dtype0*
valueB:
�
generator/strided_slice_1StridedSlicegenerator/Shape_1generator/strided_slice_1/stack!generator/strided_slice_1/stack_1!generator/strided_slice_1/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
9
generator/mul/yConst*
dtype0*
value	B :
G
generator/mulMulgenerator/strided_slicegenerator/mul/y*
T0
;
generator/mul_1/yConst*
dtype0*
value	B :
M
generator/mul_1Mulgenerator/strided_slice_1generator/mul_1/y*
T0
c
generator/ResizeBilinear/sizePackgenerator/mulgenerator/mul_1*
N*
T0*

axis 
�
generator/ResizeBilinearResizeBilineargenerator/LeakyRelu_5generator/ResizeBilinear/size*
T0*
align_corners( *
half_pixel_centers( 
P
generator/addAddV2generator/ResizeBilineargenerator/LeakyRelu_2*
T0
�
9generator/Conv_6/weights/Initializer/random_uniform/shapeConst*+
_class!
loc:@generator/Conv_6/weights*
dtype0*%
valueB"      @   @   
�
7generator/Conv_6/weights/Initializer/random_uniform/minConst*+
_class!
loc:@generator/Conv_6/weights*
dtype0*
valueB
 *:͓�
�
7generator/Conv_6/weights/Initializer/random_uniform/maxConst*+
_class!
loc:@generator/Conv_6/weights*
dtype0*
valueB
 *:͓=
�
Agenerator/Conv_6/weights/Initializer/random_uniform/RandomUniformRandomUniform9generator/Conv_6/weights/Initializer/random_uniform/shape*
T0*+
_class!
loc:@generator/Conv_6/weights*
dtype0*
seed2 *

seed 
�
7generator/Conv_6/weights/Initializer/random_uniform/subSub7generator/Conv_6/weights/Initializer/random_uniform/max7generator/Conv_6/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@generator/Conv_6/weights
�
7generator/Conv_6/weights/Initializer/random_uniform/mulMulAgenerator/Conv_6/weights/Initializer/random_uniform/RandomUniform7generator/Conv_6/weights/Initializer/random_uniform/sub*
T0*+
_class!
loc:@generator/Conv_6/weights
�
3generator/Conv_6/weights/Initializer/random_uniformAddV27generator/Conv_6/weights/Initializer/random_uniform/mul7generator/Conv_6/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@generator/Conv_6/weights
�
generator/Conv_6/weightsVarHandleOp*+
_class!
loc:@generator/Conv_6/weights*
allowed_devices
 *
	container *
dtype0*
shape:@@*)
shared_namegenerator/Conv_6/weights
i
9generator/Conv_6/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv_6/weights
�
generator/Conv_6/weights/AssignAssignVariableOpgenerator/Conv_6/weights3generator/Conv_6/weights/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
e
,generator/Conv_6/weights/Read/ReadVariableOpReadVariableOpgenerator/Conv_6/weights*
dtype0
�
)generator/Conv_6/biases/Initializer/zerosConst**
_class 
loc:@generator/Conv_6/biases*
dtype0*
valueB@*    
�
generator/Conv_6/biasesVarHandleOp**
_class 
loc:@generator/Conv_6/biases*
allowed_devices
 *
	container *
dtype0*
shape:@*(
shared_namegenerator/Conv_6/biases
g
8generator/Conv_6/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv_6/biases
�
generator/Conv_6/biases/AssignAssignVariableOpgenerator/Conv_6/biases)generator/Conv_6/biases/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
c
+generator/Conv_6/biases/Read/ReadVariableOpReadVariableOpgenerator/Conv_6/biases*
dtype0
_
&generator/Conv_6/Conv2D/ReadVariableOpReadVariableOpgenerator/Conv_6/weights*
dtype0
�
generator/Conv_6/Conv2DConv2Dgenerator/add&generator/Conv_6/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
_
'generator/Conv_6/BiasAdd/ReadVariableOpReadVariableOpgenerator/Conv_6/biases*
dtype0
�
generator/Conv_6/BiasAddBiasAddgenerator/Conv_6/Conv2D'generator/Conv_6/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
U
generator/LeakyRelu_6	LeakyRelugenerator/Conv_6/BiasAdd*
T0*
alpha%��L>
�
9generator/Conv_7/weights/Initializer/random_uniform/shapeConst*+
_class!
loc:@generator/Conv_7/weights*
dtype0*%
valueB"      @       
�
7generator/Conv_7/weights/Initializer/random_uniform/minConst*+
_class!
loc:@generator/Conv_7/weights*
dtype0*
valueB
 *����
�
7generator/Conv_7/weights/Initializer/random_uniform/maxConst*+
_class!
loc:@generator/Conv_7/weights*
dtype0*
valueB
 *���=
�
Agenerator/Conv_7/weights/Initializer/random_uniform/RandomUniformRandomUniform9generator/Conv_7/weights/Initializer/random_uniform/shape*
T0*+
_class!
loc:@generator/Conv_7/weights*
dtype0*
seed2 *

seed 
�
7generator/Conv_7/weights/Initializer/random_uniform/subSub7generator/Conv_7/weights/Initializer/random_uniform/max7generator/Conv_7/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@generator/Conv_7/weights
�
7generator/Conv_7/weights/Initializer/random_uniform/mulMulAgenerator/Conv_7/weights/Initializer/random_uniform/RandomUniform7generator/Conv_7/weights/Initializer/random_uniform/sub*
T0*+
_class!
loc:@generator/Conv_7/weights
�
3generator/Conv_7/weights/Initializer/random_uniformAddV27generator/Conv_7/weights/Initializer/random_uniform/mul7generator/Conv_7/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@generator/Conv_7/weights
�
generator/Conv_7/weightsVarHandleOp*+
_class!
loc:@generator/Conv_7/weights*
allowed_devices
 *
	container *
dtype0*
shape:@ *)
shared_namegenerator/Conv_7/weights
i
9generator/Conv_7/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv_7/weights
�
generator/Conv_7/weights/AssignAssignVariableOpgenerator/Conv_7/weights3generator/Conv_7/weights/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
e
,generator/Conv_7/weights/Read/ReadVariableOpReadVariableOpgenerator/Conv_7/weights*
dtype0
�
)generator/Conv_7/biases/Initializer/zerosConst**
_class 
loc:@generator/Conv_7/biases*
dtype0*
valueB *    
�
generator/Conv_7/biasesVarHandleOp**
_class 
loc:@generator/Conv_7/biases*
allowed_devices
 *
	container *
dtype0*
shape: *(
shared_namegenerator/Conv_7/biases
g
8generator/Conv_7/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv_7/biases
�
generator/Conv_7/biases/AssignAssignVariableOpgenerator/Conv_7/biases)generator/Conv_7/biases/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
c
+generator/Conv_7/biases/Read/ReadVariableOpReadVariableOpgenerator/Conv_7/biases*
dtype0
_
&generator/Conv_7/Conv2D/ReadVariableOpReadVariableOpgenerator/Conv_7/weights*
dtype0
�
generator/Conv_7/Conv2DConv2Dgenerator/LeakyRelu_6&generator/Conv_7/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
_
'generator/Conv_7/BiasAdd/ReadVariableOpReadVariableOpgenerator/Conv_7/biases*
dtype0
�
generator/Conv_7/BiasAddBiasAddgenerator/Conv_7/Conv2D'generator/Conv_7/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
U
generator/LeakyRelu_7	LeakyRelugenerator/Conv_7/BiasAdd*
T0*
alpha%��L>
X
generator/Shape_2Shapegenerator/LeakyRelu_7*
T0*
out_type0:��
M
generator/strided_slice_2/stackConst*
dtype0*
valueB:
O
!generator/strided_slice_2/stack_1Const*
dtype0*
valueB:
O
!generator/strided_slice_2/stack_2Const*
dtype0*
valueB:
�
generator/strided_slice_2StridedSlicegenerator/Shape_2generator/strided_slice_2/stack!generator/strided_slice_2/stack_1!generator/strided_slice_2/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
X
generator/Shape_3Shapegenerator/LeakyRelu_7*
T0*
out_type0:��
M
generator/strided_slice_3/stackConst*
dtype0*
valueB:
O
!generator/strided_slice_3/stack_1Const*
dtype0*
valueB:
O
!generator/strided_slice_3/stack_2Const*
dtype0*
valueB:
�
generator/strided_slice_3StridedSlicegenerator/Shape_3generator/strided_slice_3/stack!generator/strided_slice_3/stack_1!generator/strided_slice_3/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
;
generator/mul_2/yConst*
dtype0*
value	B :
M
generator/mul_2Mulgenerator/strided_slice_2generator/mul_2/y*
T0
;
generator/mul_3/yConst*
dtype0*
value	B :
M
generator/mul_3Mulgenerator/strided_slice_3generator/mul_3/y*
T0
g
generator/ResizeBilinear_1/sizePackgenerator/mul_2generator/mul_3*
N*
T0*

axis 
�
generator/ResizeBilinear_1ResizeBilineargenerator/LeakyRelu_7generator/ResizeBilinear_1/size*
T0*
align_corners( *
half_pixel_centers( 
R
generator/add_1AddV2generator/ResizeBilinear_1generator/LeakyRelu*
T0
�
9generator/Conv_8/weights/Initializer/random_uniform/shapeConst*+
_class!
loc:@generator/Conv_8/weights*
dtype0*%
valueB"              
�
7generator/Conv_8/weights/Initializer/random_uniform/minConst*+
_class!
loc:@generator/Conv_8/weights*
dtype0*
valueB
 *�ѽ
�
7generator/Conv_8/weights/Initializer/random_uniform/maxConst*+
_class!
loc:@generator/Conv_8/weights*
dtype0*
valueB
 *��=
�
Agenerator/Conv_8/weights/Initializer/random_uniform/RandomUniformRandomUniform9generator/Conv_8/weights/Initializer/random_uniform/shape*
T0*+
_class!
loc:@generator/Conv_8/weights*
dtype0*
seed2 *

seed 
�
7generator/Conv_8/weights/Initializer/random_uniform/subSub7generator/Conv_8/weights/Initializer/random_uniform/max7generator/Conv_8/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@generator/Conv_8/weights
�
7generator/Conv_8/weights/Initializer/random_uniform/mulMulAgenerator/Conv_8/weights/Initializer/random_uniform/RandomUniform7generator/Conv_8/weights/Initializer/random_uniform/sub*
T0*+
_class!
loc:@generator/Conv_8/weights
�
3generator/Conv_8/weights/Initializer/random_uniformAddV27generator/Conv_8/weights/Initializer/random_uniform/mul7generator/Conv_8/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@generator/Conv_8/weights
�
generator/Conv_8/weightsVarHandleOp*+
_class!
loc:@generator/Conv_8/weights*
allowed_devices
 *
	container *
dtype0*
shape:  *)
shared_namegenerator/Conv_8/weights
i
9generator/Conv_8/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv_8/weights
�
generator/Conv_8/weights/AssignAssignVariableOpgenerator/Conv_8/weights3generator/Conv_8/weights/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
e
,generator/Conv_8/weights/Read/ReadVariableOpReadVariableOpgenerator/Conv_8/weights*
dtype0
�
)generator/Conv_8/biases/Initializer/zerosConst**
_class 
loc:@generator/Conv_8/biases*
dtype0*
valueB *    
�
generator/Conv_8/biasesVarHandleOp**
_class 
loc:@generator/Conv_8/biases*
allowed_devices
 *
	container *
dtype0*
shape: *(
shared_namegenerator/Conv_8/biases
g
8generator/Conv_8/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv_8/biases
�
generator/Conv_8/biases/AssignAssignVariableOpgenerator/Conv_8/biases)generator/Conv_8/biases/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
c
+generator/Conv_8/biases/Read/ReadVariableOpReadVariableOpgenerator/Conv_8/biases*
dtype0
_
&generator/Conv_8/Conv2D/ReadVariableOpReadVariableOpgenerator/Conv_8/weights*
dtype0
�
generator/Conv_8/Conv2DConv2Dgenerator/add_1&generator/Conv_8/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
_
'generator/Conv_8/BiasAdd/ReadVariableOpReadVariableOpgenerator/Conv_8/biases*
dtype0
�
generator/Conv_8/BiasAddBiasAddgenerator/Conv_8/Conv2D'generator/Conv_8/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
U
generator/LeakyRelu_8	LeakyRelugenerator/Conv_8/BiasAdd*
T0*
alpha%��L>
�
9generator/Conv_9/weights/Initializer/random_uniform/shapeConst*+
_class!
loc:@generator/Conv_9/weights*
dtype0*%
valueB"             
�
7generator/Conv_9/weights/Initializer/random_uniform/minConst*+
_class!
loc:@generator/Conv_9/weights*
dtype0*
valueB
 *�Er�
�
7generator/Conv_9/weights/Initializer/random_uniform/maxConst*+
_class!
loc:@generator/Conv_9/weights*
dtype0*
valueB
 *�Er=
�
Agenerator/Conv_9/weights/Initializer/random_uniform/RandomUniformRandomUniform9generator/Conv_9/weights/Initializer/random_uniform/shape*
T0*+
_class!
loc:@generator/Conv_9/weights*
dtype0*
seed2 *

seed 
�
7generator/Conv_9/weights/Initializer/random_uniform/subSub7generator/Conv_9/weights/Initializer/random_uniform/max7generator/Conv_9/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@generator/Conv_9/weights
�
7generator/Conv_9/weights/Initializer/random_uniform/mulMulAgenerator/Conv_9/weights/Initializer/random_uniform/RandomUniform7generator/Conv_9/weights/Initializer/random_uniform/sub*
T0*+
_class!
loc:@generator/Conv_9/weights
�
3generator/Conv_9/weights/Initializer/random_uniformAddV27generator/Conv_9/weights/Initializer/random_uniform/mul7generator/Conv_9/weights/Initializer/random_uniform/min*
T0*+
_class!
loc:@generator/Conv_9/weights
�
generator/Conv_9/weightsVarHandleOp*+
_class!
loc:@generator/Conv_9/weights*
allowed_devices
 *
	container *
dtype0*
shape: *)
shared_namegenerator/Conv_9/weights
i
9generator/Conv_9/weights/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv_9/weights
�
generator/Conv_9/weights/AssignAssignVariableOpgenerator/Conv_9/weights3generator/Conv_9/weights/Initializer/random_uniform*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
e
,generator/Conv_9/weights/Read/ReadVariableOpReadVariableOpgenerator/Conv_9/weights*
dtype0
�
)generator/Conv_9/biases/Initializer/zerosConst**
_class 
loc:@generator/Conv_9/biases*
dtype0*
valueB*    
�
generator/Conv_9/biasesVarHandleOp**
_class 
loc:@generator/Conv_9/biases*
allowed_devices
 *
	container *
dtype0*
shape:*(
shared_namegenerator/Conv_9/biases
g
8generator/Conv_9/biases/IsInitialized/VarIsInitializedOpVarIsInitializedOpgenerator/Conv_9/biases
�
generator/Conv_9/biases/AssignAssignVariableOpgenerator/Conv_9/biases)generator/Conv_9/biases/Initializer/zeros*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
c
+generator/Conv_9/biases/Read/ReadVariableOpReadVariableOpgenerator/Conv_9/biases*
dtype0
_
&generator/Conv_9/Conv2D/ReadVariableOpReadVariableOpgenerator/Conv_9/weights*
dtype0
�
generator/Conv_9/Conv2DConv2Dgenerator/LeakyRelu_8&generator/Conv_9/Conv2D/ReadVariableOp*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides
*
use_cudnn_on_gpu(
_
'generator/Conv_9/BiasAdd/ReadVariableOpReadVariableOpgenerator/Conv_9/biases*
dtype0
�
generator/Conv_9/BiasAddBiasAddgenerator/Conv_9/Conv2D'generator/Conv_9/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
B
ShapeShapeinput_photo*
T0*
out_type0:��
A
strided_slice/stackConst*
dtype0*
valueB:
C
strided_slice/stack_1Const*
dtype0*
valueB:
C
strided_slice/stack_2Const*
dtype0*
valueB:
�
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
C
strided_slice_1/stackConst*
dtype0*
valueB:
E
strided_slice_1/stack_1Const*
dtype0*
valueB:
E
strided_slice_1/stack_2Const*
dtype0*
valueB:
�
strided_slice_1StridedSliceShapestrided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
7
ones/packed/0Const*
dtype0*
value	B :
7
ones/packed/3Const*
dtype0*
value	B :
o
ones/packedPackones/packed/0strided_slicestrided_slice_1ones/packed/3*
N*
T0*

axis 
7

ones/ConstConst*
dtype0*
valueB
 *  �?
@
onesFillones/packed
ones/Const*
T0*

index_type0
p
depthwise/filter_inConst*
dtype0*E
value<B:"$9��=9��=9��=9��=9��=9��=9��=9��=9��=
L
depthwise/ShapeConst*
dtype0*%
valueB"            
L
depthwise/dilation_rateConst*
dtype0*
valueB"      
�
	depthwiseDepthwiseConv2dNativeonesdepthwise/filter_in*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides

�
depthwise_1/filter_inConst*
dtype0*�
value�B�"l9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=
N
depthwise_1/ShapeConst*
dtype0*%
valueB"            
N
depthwise_1/dilation_rateConst*
dtype0*
valueB"      
�
depthwise_1DepthwiseConv2dNativeinput_photodepthwise_1/filter_in*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides

3
truedivRealDivdepthwise_1	depthwise*
T0
�
depthwise_2/filter_inConst*
dtype0*�
value�B�"l9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=
N
depthwise_2/ShapeConst*
dtype0*%
valueB"            
N
depthwise_2/dilation_rateConst*
dtype0*
valueB"      
�
depthwise_2DepthwiseConv2dNativegenerator/Conv_9/BiasAdddepthwise_2/filter_in*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides

5
	truediv_1RealDivdepthwise_2	depthwise*
T0
:
mulMulinput_photogenerator/Conv_9/BiasAdd*
T0
�
depthwise_3/filter_inConst*
dtype0*�
value�B�"l9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=
N
depthwise_3/ShapeConst*
dtype0*%
valueB"            
N
depthwise_3/dilation_rateConst*
dtype0*
valueB"      
�
depthwise_3DepthwiseConv2dNativemuldepthwise_3/filter_in*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides

5
	truediv_2RealDivdepthwise_3	depthwise*
T0
)
mul_1Multruediv	truediv_1*
T0
%
subSub	truediv_2mul_1*
T0
/
mul_2Mulinput_photoinput_photo*
T0
�
depthwise_4/filter_inConst*
dtype0*�
value�B�"l9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=
N
depthwise_4/ShapeConst*
dtype0*%
valueB"            
N
depthwise_4/dilation_rateConst*
dtype0*
valueB"      
�
depthwise_4DepthwiseConv2dNativemul_2depthwise_4/filter_in*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides

5
	truediv_3RealDivdepthwise_4	depthwise*
T0
'
mul_3Multruedivtruediv*
T0
'
sub_1Sub	truediv_3mul_3*
T0
2
add/yConst*
dtype0*
valueB
 *
ף;
#
addAddV2sub_1add/y*
T0
'
	truediv_4RealDivsubadd*
T0
)
mul_4Mul	truediv_4truediv*
T0
'
sub_2Sub	truediv_1mul_4*
T0
�
depthwise_5/filter_inConst*
dtype0*�
value�B�"l9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=
N
depthwise_5/ShapeConst*
dtype0*%
valueB"            
N
depthwise_5/dilation_rateConst*
dtype0*
valueB"      
�
depthwise_5DepthwiseConv2dNative	truediv_4depthwise_5/filter_in*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides

5
	truediv_5RealDivdepthwise_5	depthwise*
T0
�
depthwise_6/filter_inConst*
dtype0*�
value�B�"l9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=9��=
N
depthwise_6/ShapeConst*
dtype0*%
valueB"            
N
depthwise_6/dilation_rateConst*
dtype0*
valueB"      
�
depthwise_6DepthwiseConv2dNativesub_2depthwise_6/filter_in*
T0*
data_formatNHWC*
	dilations
*
explicit_paddings
 *
paddingSAME*
strides

5
	truediv_6RealDivdepthwise_6	depthwise*
T0
-
mul_5Mul	truediv_5input_photo*
T0
)
add_1AddV2mul_5	truediv_6*
T0
"
outputIdentityadd_1*
T0
A
save/filename/inputConst*
dtype0*
valueB Bmodel
V
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
shape: 
M

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
shape: 
�
save/SaveV2/tensor_namesConst*
dtype0*�
value�B�$Bgenerator/Conv/biasesBgenerator/Conv/weightsBgenerator/Conv_1/biasesBgenerator/Conv_1/weightsBgenerator/Conv_2/biasesBgenerator/Conv_2/weightsBgenerator/Conv_3/biasesBgenerator/Conv_3/weightsBgenerator/Conv_4/biasesBgenerator/Conv_4/weightsBgenerator/Conv_5/biasesBgenerator/Conv_5/weightsBgenerator/Conv_6/biasesBgenerator/Conv_6/weightsBgenerator/Conv_7/biasesBgenerator/Conv_7/weightsBgenerator/Conv_8/biasesBgenerator/Conv_8/weightsBgenerator/Conv_9/biasesBgenerator/Conv_9/weightsBgenerator/block_0/conv1/biasesBgenerator/block_0/conv1/weightsBgenerator/block_0/conv2/biasesBgenerator/block_0/conv2/weightsBgenerator/block_1/conv1/biasesBgenerator/block_1/conv1/weightsBgenerator/block_1/conv2/biasesBgenerator/block_1/conv2/weightsBgenerator/block_2/conv1/biasesBgenerator/block_2/conv1/weightsBgenerator/block_2/conv2/biasesBgenerator/block_2/conv2/weightsBgenerator/block_3/conv1/biasesBgenerator/block_3/conv1/weightsBgenerator/block_3/conv2/biasesBgenerator/block_3/conv2/weights
�
save/SaveV2/shape_and_slicesConst*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slices)generator/Conv/biases/Read/ReadVariableOp*generator/Conv/weights/Read/ReadVariableOp+generator/Conv_1/biases/Read/ReadVariableOp,generator/Conv_1/weights/Read/ReadVariableOp+generator/Conv_2/biases/Read/ReadVariableOp,generator/Conv_2/weights/Read/ReadVariableOp+generator/Conv_3/biases/Read/ReadVariableOp,generator/Conv_3/weights/Read/ReadVariableOp+generator/Conv_4/biases/Read/ReadVariableOp,generator/Conv_4/weights/Read/ReadVariableOp+generator/Conv_5/biases/Read/ReadVariableOp,generator/Conv_5/weights/Read/ReadVariableOp+generator/Conv_6/biases/Read/ReadVariableOp,generator/Conv_6/weights/Read/ReadVariableOp+generator/Conv_7/biases/Read/ReadVariableOp,generator/Conv_7/weights/Read/ReadVariableOp+generator/Conv_8/biases/Read/ReadVariableOp,generator/Conv_8/weights/Read/ReadVariableOp+generator/Conv_9/biases/Read/ReadVariableOp,generator/Conv_9/weights/Read/ReadVariableOp2generator/block_0/conv1/biases/Read/ReadVariableOp3generator/block_0/conv1/weights/Read/ReadVariableOp2generator/block_0/conv2/biases/Read/ReadVariableOp3generator/block_0/conv2/weights/Read/ReadVariableOp2generator/block_1/conv1/biases/Read/ReadVariableOp3generator/block_1/conv1/weights/Read/ReadVariableOp2generator/block_1/conv2/biases/Read/ReadVariableOp3generator/block_1/conv2/weights/Read/ReadVariableOp2generator/block_2/conv1/biases/Read/ReadVariableOp3generator/block_2/conv1/weights/Read/ReadVariableOp2generator/block_2/conv2/biases/Read/ReadVariableOp3generator/block_2/conv2/weights/Read/ReadVariableOp2generator/block_3/conv1/biases/Read/ReadVariableOp3generator/block_3/conv1/weights/Read/ReadVariableOp2generator/block_3/conv2/biases/Read/ReadVariableOp3generator/block_3/conv2/weights/Read/ReadVariableOp*&
 _has_manual_control_dependencies(*2
dtypes(
&2$
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*�
value�B�$Bgenerator/Conv/biasesBgenerator/Conv/weightsBgenerator/Conv_1/biasesBgenerator/Conv_1/weightsBgenerator/Conv_2/biasesBgenerator/Conv_2/weightsBgenerator/Conv_3/biasesBgenerator/Conv_3/weightsBgenerator/Conv_4/biasesBgenerator/Conv_4/weightsBgenerator/Conv_5/biasesBgenerator/Conv_5/weightsBgenerator/Conv_6/biasesBgenerator/Conv_6/weightsBgenerator/Conv_7/biasesBgenerator/Conv_7/weightsBgenerator/Conv_8/biasesBgenerator/Conv_8/weightsBgenerator/Conv_9/biasesBgenerator/Conv_9/weightsBgenerator/block_0/conv1/biasesBgenerator/block_0/conv1/weightsBgenerator/block_0/conv2/biasesBgenerator/block_0/conv2/weightsBgenerator/block_1/conv1/biasesBgenerator/block_1/conv1/weightsBgenerator/block_1/conv2/biasesBgenerator/block_1/conv2/weightsBgenerator/block_2/conv1/biasesBgenerator/block_2/conv1/weightsBgenerator/block_2/conv2/biasesBgenerator/block_2/conv2/weightsBgenerator/block_3/conv1/biasesBgenerator/block_3/conv1/weightsBgenerator/block_3/conv2/biasesBgenerator/block_3/conv2/weights
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*2
dtypes(
&2$
2
save/IdentityIdentitysave/RestoreV2*
T0
�
save/AssignVariableOpAssignVariableOpgenerator/Conv/biasessave/Identity*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
6
save/Identity_1Identitysave/RestoreV2:1*
T0
�
save/AssignVariableOp_1AssignVariableOpgenerator/Conv/weightssave/Identity_1*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
6
save/Identity_2Identitysave/RestoreV2:2*
T0
�
save/AssignVariableOp_2AssignVariableOpgenerator/Conv_1/biasessave/Identity_2*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
6
save/Identity_3Identitysave/RestoreV2:3*
T0
�
save/AssignVariableOp_3AssignVariableOpgenerator/Conv_1/weightssave/Identity_3*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
6
save/Identity_4Identitysave/RestoreV2:4*
T0
�
save/AssignVariableOp_4AssignVariableOpgenerator/Conv_2/biasessave/Identity_4*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
6
save/Identity_5Identitysave/RestoreV2:5*
T0
�
save/AssignVariableOp_5AssignVariableOpgenerator/Conv_2/weightssave/Identity_5*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
6
save/Identity_6Identitysave/RestoreV2:6*
T0
�
save/AssignVariableOp_6AssignVariableOpgenerator/Conv_3/biasessave/Identity_6*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
6
save/Identity_7Identitysave/RestoreV2:7*
T0
�
save/AssignVariableOp_7AssignVariableOpgenerator/Conv_3/weightssave/Identity_7*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
6
save/Identity_8Identitysave/RestoreV2:8*
T0
�
save/AssignVariableOp_8AssignVariableOpgenerator/Conv_4/biasessave/Identity_8*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
6
save/Identity_9Identitysave/RestoreV2:9*
T0
�
save/AssignVariableOp_9AssignVariableOpgenerator/Conv_4/weightssave/Identity_9*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_10Identitysave/RestoreV2:10*
T0
�
save/AssignVariableOp_10AssignVariableOpgenerator/Conv_5/biasessave/Identity_10*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_11Identitysave/RestoreV2:11*
T0
�
save/AssignVariableOp_11AssignVariableOpgenerator/Conv_5/weightssave/Identity_11*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_12Identitysave/RestoreV2:12*
T0
�
save/AssignVariableOp_12AssignVariableOpgenerator/Conv_6/biasessave/Identity_12*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_13Identitysave/RestoreV2:13*
T0
�
save/AssignVariableOp_13AssignVariableOpgenerator/Conv_6/weightssave/Identity_13*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_14Identitysave/RestoreV2:14*
T0
�
save/AssignVariableOp_14AssignVariableOpgenerator/Conv_7/biasessave/Identity_14*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_15Identitysave/RestoreV2:15*
T0
�
save/AssignVariableOp_15AssignVariableOpgenerator/Conv_7/weightssave/Identity_15*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_16Identitysave/RestoreV2:16*
T0
�
save/AssignVariableOp_16AssignVariableOpgenerator/Conv_8/biasessave/Identity_16*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_17Identitysave/RestoreV2:17*
T0
�
save/AssignVariableOp_17AssignVariableOpgenerator/Conv_8/weightssave/Identity_17*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_18Identitysave/RestoreV2:18*
T0
�
save/AssignVariableOp_18AssignVariableOpgenerator/Conv_9/biasessave/Identity_18*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_19Identitysave/RestoreV2:19*
T0
�
save/AssignVariableOp_19AssignVariableOpgenerator/Conv_9/weightssave/Identity_19*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_20Identitysave/RestoreV2:20*
T0
�
save/AssignVariableOp_20AssignVariableOpgenerator/block_0/conv1/biasessave/Identity_20*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_21Identitysave/RestoreV2:21*
T0
�
save/AssignVariableOp_21AssignVariableOpgenerator/block_0/conv1/weightssave/Identity_21*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_22Identitysave/RestoreV2:22*
T0
�
save/AssignVariableOp_22AssignVariableOpgenerator/block_0/conv2/biasessave/Identity_22*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_23Identitysave/RestoreV2:23*
T0
�
save/AssignVariableOp_23AssignVariableOpgenerator/block_0/conv2/weightssave/Identity_23*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_24Identitysave/RestoreV2:24*
T0
�
save/AssignVariableOp_24AssignVariableOpgenerator/block_1/conv1/biasessave/Identity_24*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_25Identitysave/RestoreV2:25*
T0
�
save/AssignVariableOp_25AssignVariableOpgenerator/block_1/conv1/weightssave/Identity_25*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_26Identitysave/RestoreV2:26*
T0
�
save/AssignVariableOp_26AssignVariableOpgenerator/block_1/conv2/biasessave/Identity_26*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_27Identitysave/RestoreV2:27*
T0
�
save/AssignVariableOp_27AssignVariableOpgenerator/block_1/conv2/weightssave/Identity_27*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_28Identitysave/RestoreV2:28*
T0
�
save/AssignVariableOp_28AssignVariableOpgenerator/block_2/conv1/biasessave/Identity_28*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_29Identitysave/RestoreV2:29*
T0
�
save/AssignVariableOp_29AssignVariableOpgenerator/block_2/conv1/weightssave/Identity_29*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_30Identitysave/RestoreV2:30*
T0
�
save/AssignVariableOp_30AssignVariableOpgenerator/block_2/conv2/biasessave/Identity_30*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_31Identitysave/RestoreV2:31*
T0
�
save/AssignVariableOp_31AssignVariableOpgenerator/block_2/conv2/weightssave/Identity_31*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_32Identitysave/RestoreV2:32*
T0
�
save/AssignVariableOp_32AssignVariableOpgenerator/block_3/conv1/biasessave/Identity_32*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_33Identitysave/RestoreV2:33*
T0
�
save/AssignVariableOp_33AssignVariableOpgenerator/block_3/conv1/weightssave/Identity_33*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_34Identitysave/RestoreV2:34*
T0
�
save/AssignVariableOp_34AssignVariableOpgenerator/block_3/conv2/biasessave/Identity_34*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
8
save/Identity_35Identitysave/RestoreV2:35*
T0
�
save/AssignVariableOp_35AssignVariableOpgenerator/block_3/conv2/weightssave/Identity_35*&
 _has_manual_control_dependencies(*
dtype0*
validate_shape( 
�
save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_35^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
�

initNoOp^generator/Conv/biases/Assign^generator/Conv/weights/Assign^generator/Conv_1/biases/Assign ^generator/Conv_1/weights/Assign^generator/Conv_2/biases/Assign ^generator/Conv_2/weights/Assign^generator/Conv_3/biases/Assign ^generator/Conv_3/weights/Assign^generator/Conv_4/biases/Assign ^generator/Conv_4/weights/Assign^generator/Conv_5/biases/Assign ^generator/Conv_5/weights/Assign^generator/Conv_6/biases/Assign ^generator/Conv_6/weights/Assign^generator/Conv_7/biases/Assign ^generator/Conv_7/weights/Assign^generator/Conv_8/biases/Assign ^generator/Conv_8/weights/Assign^generator/Conv_9/biases/Assign ^generator/Conv_9/weights/Assign&^generator/block_0/conv1/biases/Assign'^generator/block_0/conv1/weights/Assign&^generator/block_0/conv2/biases/Assign'^generator/block_0/conv2/weights/Assign&^generator/block_1/conv1/biases/Assign'^generator/block_1/conv1/weights/Assign&^generator/block_1/conv2/biases/Assign'^generator/block_1/conv2/weights/Assign&^generator/block_2/conv1/biases/Assign'^generator/block_2/conv1/weights/Assign&^generator/block_2/conv2/biases/Assign'^generator/block_2/conv2/weights/Assign&^generator/block_3/conv1/biases/Assign'^generator/block_3/conv1/weights/Assign&^generator/block_3/conv2/biases/Assign'^generator/block_3/conv2/weights/Assign"�