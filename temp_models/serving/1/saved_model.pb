��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02v2.1.0-rc2-17-ge5bf8de4108��
{
dense_84/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	0�* 
shared_namedense_84/kernel
t
#dense_84/kernel/Read/ReadVariableOpReadVariableOpdense_84/kernel*
_output_shapes
:	0�*
dtype0
s
dense_84/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_84/bias
l
!dense_84/bias/Read/ReadVariableOpReadVariableOpdense_84/bias*
_output_shapes	
:�*
dtype0
|
dense_85/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��* 
shared_namedense_85/kernel
u
#dense_85/kernel/Read/ReadVariableOpReadVariableOpdense_85/kernel* 
_output_shapes
:
��*
dtype0
s
dense_85/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_85/bias
l
!dense_85/bias/Read/ReadVariableOpReadVariableOpdense_85/bias*
_output_shapes	
:�*
dtype0
{
dense_86/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_86/kernel
t
#dense_86/kernel/Read/ReadVariableOpReadVariableOpdense_86/kernel*
_output_shapes
:	�*
dtype0
r
dense_86/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_86/bias
k
!dense_86/bias/Read/ReadVariableOpReadVariableOpdense_86/bias*
_output_shapes
:*
dtype0
x
training/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *#
shared_nametraining/Adam/iter
q
&training/Adam/iter/Read/ReadVariableOpReadVariableOptraining/Adam/iter*
_output_shapes
: *
dtype0	
|
training/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nametraining/Adam/beta_1
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
_output_shapes
: *
dtype0
|
training/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nametraining/Adam/beta_2
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
_output_shapes
: *
dtype0
z
training/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nametraining/Adam/decay
s
'training/Adam/decay/Read/ReadVariableOpReadVariableOptraining/Adam/decay*
_output_shapes
: *
dtype0
�
training/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nametraining/Adam/learning_rate
�
/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
_output_shapes
: *
dtype0
w
gru_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namegru_28/kernel
p
!gru_28/kernel/Read/ReadVariableOpReadVariableOpgru_28/kernel*
_output_shapes
:	�*
dtype0
�
gru_28/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	0�*(
shared_namegru_28/recurrent_kernel
�
+gru_28/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru_28/recurrent_kernel*
_output_shapes
:	0�*
dtype0
s
gru_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namegru_28/bias
l
gru_28/bias/Read/ReadVariableOpReadVariableOpgru_28/bias*
_output_shapes
:	�*
dtype0
�
training/Adam/dense_84/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	0�*0
shared_name!training/Adam/dense_84/kernel/m
�
3training/Adam/dense_84/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_84/kernel/m*
_output_shapes
:	0�*
dtype0
�
training/Adam/dense_84/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_nametraining/Adam/dense_84/bias/m
�
1training/Adam/dense_84/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_84/bias/m*
_output_shapes	
:�*
dtype0
�
training/Adam/dense_85/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*0
shared_name!training/Adam/dense_85/kernel/m
�
3training/Adam/dense_85/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_85/kernel/m* 
_output_shapes
:
��*
dtype0
�
training/Adam/dense_85/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_nametraining/Adam/dense_85/bias/m
�
1training/Adam/dense_85/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_85/bias/m*
_output_shapes	
:�*
dtype0
�
training/Adam/dense_86/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*0
shared_name!training/Adam/dense_86/kernel/m
�
3training/Adam/dense_86/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_86/kernel/m*
_output_shapes
:	�*
dtype0
�
training/Adam/dense_86/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nametraining/Adam/dense_86/bias/m
�
1training/Adam/dense_86/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_86/bias/m*
_output_shapes
:*
dtype0
�
training/Adam/gru_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*.
shared_nametraining/Adam/gru_28/kernel/m
�
1training/Adam/gru_28/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/gru_28/kernel/m*
_output_shapes
:	�*
dtype0
�
'training/Adam/gru_28/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	0�*8
shared_name)'training/Adam/gru_28/recurrent_kernel/m
�
;training/Adam/gru_28/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp'training/Adam/gru_28/recurrent_kernel/m*
_output_shapes
:	0�*
dtype0
�
training/Adam/gru_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*,
shared_nametraining/Adam/gru_28/bias/m
�
/training/Adam/gru_28/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/gru_28/bias/m*
_output_shapes
:	�*
dtype0
�
training/Adam/dense_84/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	0�*0
shared_name!training/Adam/dense_84/kernel/v
�
3training/Adam/dense_84/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_84/kernel/v*
_output_shapes
:	0�*
dtype0
�
training/Adam/dense_84/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_nametraining/Adam/dense_84/bias/v
�
1training/Adam/dense_84/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_84/bias/v*
_output_shapes	
:�*
dtype0
�
training/Adam/dense_85/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*0
shared_name!training/Adam/dense_85/kernel/v
�
3training/Adam/dense_85/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_85/kernel/v* 
_output_shapes
:
��*
dtype0
�
training/Adam/dense_85/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_nametraining/Adam/dense_85/bias/v
�
1training/Adam/dense_85/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_85/bias/v*
_output_shapes	
:�*
dtype0
�
training/Adam/dense_86/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*0
shared_name!training/Adam/dense_86/kernel/v
�
3training/Adam/dense_86/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_86/kernel/v*
_output_shapes
:	�*
dtype0
�
training/Adam/dense_86/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nametraining/Adam/dense_86/bias/v
�
1training/Adam/dense_86/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_86/bias/v*
_output_shapes
:*
dtype0
�
training/Adam/gru_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*.
shared_nametraining/Adam/gru_28/kernel/v
�
1training/Adam/gru_28/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/gru_28/kernel/v*
_output_shapes
:	�*
dtype0
�
'training/Adam/gru_28/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	0�*8
shared_name)'training/Adam/gru_28/recurrent_kernel/v
�
;training/Adam/gru_28/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp'training/Adam/gru_28/recurrent_kernel/v*
_output_shapes
:	0�*
dtype0
�
training/Adam/gru_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*,
shared_nametraining/Adam/gru_28/bias/v
�
/training/Adam/gru_28/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/gru_28/bias/v*
_output_shapes
:	�*
dtype0

NoOpNoOp
�/
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�/
value�/B�/ B�/
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
 
l
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
�
$iter

%beta_1

&beta_2
	'decay
(learning_ratemHmImJmKmLmM)mN*mO+mPvQvRvSvTvUvV)vW*vX+vY
?
)0
*1
+2
3
4
5
6
7
8
 
?
)0
*1
+2
3
4
5
6
7
8
�
,layer_regularization_losses
	variables
-metrics
.non_trainable_variables
regularization_losses

/layers
	trainable_variables
 
~

)kernel
*recurrent_kernel
+bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
 

)0
*1
+2
 

)0
*1
+2
�
4layer_regularization_losses
	variables
5metrics
6non_trainable_variables
regularization_losses

7layers
trainable_variables
[Y
VARIABLE_VALUEdense_84/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_84/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
8layer_regularization_losses
	variables
9metrics
:non_trainable_variables
regularization_losses

;layers
trainable_variables
[Y
VARIABLE_VALUEdense_85/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_85/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
<layer_regularization_losses
	variables
=metrics
>non_trainable_variables
regularization_losses

?layers
trainable_variables
[Y
VARIABLE_VALUEdense_86/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_86/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
@layer_regularization_losses
 	variables
Ametrics
Bnon_trainable_variables
!regularization_losses

Clayers
"trainable_variables
QO
VARIABLE_VALUEtraining/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtraining/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEtraining/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEgru_28/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEgru_28/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEgru_28/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
2
3

)0
*1
+2
 

)0
*1
+2
�
Dlayer_regularization_losses
0	variables
Emetrics
Fnon_trainable_variables
1regularization_losses

Glayers
2trainable_variables
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
��
VARIABLE_VALUEtraining/Adam/dense_84/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_84/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_85/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_85/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_86/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_86/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEtraining/Adam/gru_28/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'training/Adam/gru_28/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEtraining/Adam/gru_28/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_84/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_84/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_85/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_85/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_86/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEtraining/Adam/dense_86/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEtraining/Adam/gru_28/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'training/Adam/gru_28/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEtraining/Adam/gru_28/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_gru_28_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_28_inputgru_28/biasgru_28/kernelgru_28/recurrent_kerneldense_84/kerneldense_84/biasdense_85/kerneldense_85/biasdense_86/kerneldense_86/bias*
Tin
2
*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference_signature_wrapper_2599
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_84/kernel/Read/ReadVariableOp!dense_84/bias/Read/ReadVariableOp#dense_85/kernel/Read/ReadVariableOp!dense_85/bias/Read/ReadVariableOp#dense_86/kernel/Read/ReadVariableOp!dense_86/bias/Read/ReadVariableOp&training/Adam/iter/Read/ReadVariableOp(training/Adam/beta_1/Read/ReadVariableOp(training/Adam/beta_2/Read/ReadVariableOp'training/Adam/decay/Read/ReadVariableOp/training/Adam/learning_rate/Read/ReadVariableOp!gru_28/kernel/Read/ReadVariableOp+gru_28/recurrent_kernel/Read/ReadVariableOpgru_28/bias/Read/ReadVariableOp3training/Adam/dense_84/kernel/m/Read/ReadVariableOp1training/Adam/dense_84/bias/m/Read/ReadVariableOp3training/Adam/dense_85/kernel/m/Read/ReadVariableOp1training/Adam/dense_85/bias/m/Read/ReadVariableOp3training/Adam/dense_86/kernel/m/Read/ReadVariableOp1training/Adam/dense_86/bias/m/Read/ReadVariableOp1training/Adam/gru_28/kernel/m/Read/ReadVariableOp;training/Adam/gru_28/recurrent_kernel/m/Read/ReadVariableOp/training/Adam/gru_28/bias/m/Read/ReadVariableOp3training/Adam/dense_84/kernel/v/Read/ReadVariableOp1training/Adam/dense_84/bias/v/Read/ReadVariableOp3training/Adam/dense_85/kernel/v/Read/ReadVariableOp1training/Adam/dense_85/bias/v/Read/ReadVariableOp3training/Adam/dense_86/kernel/v/Read/ReadVariableOp1training/Adam/dense_86/bias/v/Read/ReadVariableOp1training/Adam/gru_28/kernel/v/Read/ReadVariableOp;training/Adam/gru_28/recurrent_kernel/v/Read/ReadVariableOp/training/Adam/gru_28/bias/v/Read/ReadVariableOpConst*-
Tin&
$2"	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*&
f!R
__inference__traced_save_3922
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_84/kerneldense_84/biasdense_85/kerneldense_85/biasdense_86/kerneldense_86/biastraining/Adam/itertraining/Adam/beta_1training/Adam/beta_2training/Adam/decaytraining/Adam/learning_rategru_28/kernelgru_28/recurrent_kernelgru_28/biastraining/Adam/dense_84/kernel/mtraining/Adam/dense_84/bias/mtraining/Adam/dense_85/kernel/mtraining/Adam/dense_85/bias/mtraining/Adam/dense_86/kernel/mtraining/Adam/dense_86/bias/mtraining/Adam/gru_28/kernel/m'training/Adam/gru_28/recurrent_kernel/mtraining/Adam/gru_28/bias/mtraining/Adam/dense_84/kernel/vtraining/Adam/dense_84/bias/vtraining/Adam/dense_85/kernel/vtraining/Adam/dense_85/bias/vtraining/Adam/dense_86/kernel/vtraining/Adam/dense_86/bias/vtraining/Adam/gru_28/kernel/v'training/Adam/gru_28/recurrent_kernel/vtraining/Adam/gru_28/bias/v*,
Tin%
#2!*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__traced_restore_4030��
�P
�
@__inference_gru_28_layer_call_and_return_conditional_losses_3473

inputs
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :02
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :02
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������02
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanhd
mul_1MulSigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcematmul_readvariableop_resource matmul_1_readvariableop_resource^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������0: : : : : *
bodyR
while_body_3384*
condR
while_cond_3383*8
output_shapes'
%: : : : :���������0: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������0*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������0*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������02
transpose_1�
IdentityIdentitystrided_slice_3:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:���������02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs
�
�
while_cond_2158
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1,
(while_cond_2158___redundant_placeholder0,
(while_cond_2158___redundant_placeholder1,
(while_cond_2158___redundant_placeholder2,
(while_cond_2158___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :���������0: ::::
�3
�
$sequential_28_gru_28_while_body_1456+
'sequential_28_gru_28_while_loop_counter1
-sequential_28_gru_28_while_maximum_iterations
placeholder
placeholder_1
placeholder_2*
&sequential_28_gru_28_strided_slice_1_0f
btensorarrayv2read_tensorlistgetitem_sequential_28_gru_28_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4(
$sequential_28_gru_28_strided_slice_1d
`tensorarrayv2read_tensorlistgetitem_sequential_28_gru_28_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItembtensorarrayv2read_tensorlistgetitem_sequential_28_gru_28_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem{
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/ys
add_5AddV2'sequential_28_gru_28_while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5�
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identity-sequential_28_gru_28_while_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0"N
$sequential_28_gru_28_strided_slice_1&sequential_28_gru_28_strided_slice_1_0"�
`tensorarrayv2read_tensorlistgetitem_sequential_28_gru_28_tensorarrayunstack_tensorlistfromtensorbtensorarrayv2read_tensorlistgetitem_sequential_28_gru_28_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������0: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
�
�
B__inference_gru_cell_layer_call_and_return_conditional_losses_1637

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanh\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������02

Identity�

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������02

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:���������0:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
�
�
%__inference_gru_28_layer_call_fn_3315
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������0**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_gru_28_layer_call_and_return_conditional_losses_20812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������02

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
�
�
B__inference_gru_cell_layer_call_and_return_conditional_losses_1677

inputs

states
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanh\
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������02

Identity�

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������02

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:���������0:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
�
�
while_cond_2316
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1,
(while_cond_2316___redundant_placeholder0,
(while_cond_2316___redundant_placeholder1,
(while_cond_2316___redundant_placeholder2,
(while_cond_2316___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :���������0: ::::
�
�
while_cond_3209
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1,
(while_cond_3209___redundant_placeholder0,
(while_cond_3209___redundant_placeholder1,
(while_cond_3209___redundant_placeholder2,
(while_cond_3209___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :���������0: ::::
�1
�
while_body_2159
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem{
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5�
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������0: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
�1
�
while_body_2317
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem{
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5�
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������0: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
�x
�
G__inference_sequential_28_layer_call_and_return_conditional_losses_2955

inputs"
gru_28_readvariableop_resource)
%gru_28_matmul_readvariableop_resource+
'gru_28_matmul_1_readvariableop_resource+
'dense_84_matmul_readvariableop_resource,
(dense_84_biasadd_readvariableop_resource+
'dense_85_matmul_readvariableop_resource,
(dense_85_biasadd_readvariableop_resource+
'dense_86_matmul_readvariableop_resource,
(dense_86_biasadd_readvariableop_resource
identity��dense_84/BiasAdd/ReadVariableOp�dense_84/MatMul/ReadVariableOp�dense_85/BiasAdd/ReadVariableOp�dense_85/MatMul/ReadVariableOp�dense_86/BiasAdd/ReadVariableOp�dense_86/MatMul/ReadVariableOp�gru_28/MatMul/ReadVariableOp�gru_28/MatMul_1/ReadVariableOp�gru_28/ReadVariableOp�gru_28/whileR
gru_28/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_28/Shape�
gru_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_28/strided_slice/stack�
gru_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_28/strided_slice/stack_1�
gru_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_28/strided_slice/stack_2�
gru_28/strided_sliceStridedSlicegru_28/Shape:output:0#gru_28/strided_slice/stack:output:0%gru_28/strided_slice/stack_1:output:0%gru_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_28/strided_slicej
gru_28/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :02
gru_28/zeros/mul/y�
gru_28/zeros/mulMulgru_28/strided_slice:output:0gru_28/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_28/zeros/mulm
gru_28/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
gru_28/zeros/Less/y�
gru_28/zeros/LessLessgru_28/zeros/mul:z:0gru_28/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_28/zeros/Lessp
gru_28/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :02
gru_28/zeros/packed/1�
gru_28/zeros/packedPackgru_28/strided_slice:output:0gru_28/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_28/zeros/packedm
gru_28/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_28/zeros/Const�
gru_28/zerosFillgru_28/zeros/packed:output:0gru_28/zeros/Const:output:0*
T0*'
_output_shapes
:���������02
gru_28/zeros�
gru_28/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_28/transpose/perm�
gru_28/transpose	Transposeinputsgru_28/transpose/perm:output:0*
T0*+
_output_shapes
:���������2
gru_28/transposed
gru_28/Shape_1Shapegru_28/transpose:y:0*
T0*
_output_shapes
:2
gru_28/Shape_1�
gru_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_28/strided_slice_1/stack�
gru_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_1/stack_1�
gru_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_1/stack_2�
gru_28/strided_slice_1StridedSlicegru_28/Shape_1:output:0%gru_28/strided_slice_1/stack:output:0'gru_28/strided_slice_1/stack_1:output:0'gru_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_28/strided_slice_1�
"gru_28/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"gru_28/TensorArrayV2/element_shape�
gru_28/TensorArrayV2TensorListReserve+gru_28/TensorArrayV2/element_shape:output:0gru_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_28/TensorArrayV2�
<gru_28/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2>
<gru_28/TensorArrayUnstack/TensorListFromTensor/element_shape�
.gru_28/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_28/transpose:y:0Egru_28/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_28/TensorArrayUnstack/TensorListFromTensor�
gru_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_28/strided_slice_2/stack�
gru_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_2/stack_1�
gru_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_2/stack_2�
gru_28/strided_slice_2StridedSlicegru_28/transpose:y:0%gru_28/strided_slice_2/stack:output:0'gru_28/strided_slice_2/stack_1:output:0'gru_28/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
gru_28/strided_slice_2�
gru_28/ReadVariableOpReadVariableOpgru_28_readvariableop_resource*
_output_shapes
:	�*
dtype02
gru_28/ReadVariableOp�
gru_28/unstackUnpackgru_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
gru_28/unstack�
gru_28/MatMul/ReadVariableOpReadVariableOp%gru_28_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
gru_28/MatMul/ReadVariableOp�
gru_28/MatMulMatMulgru_28/strided_slice_2:output:0$gru_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_28/MatMul�
gru_28/BiasAddBiasAddgru_28/MatMul:product:0gru_28/unstack:output:0*
T0*(
_output_shapes
:����������2
gru_28/BiasAdd^
gru_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_28/Const{
gru_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_28/split/split_dim�
gru_28/splitSplitgru_28/split/split_dim:output:0gru_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
gru_28/split�
gru_28/MatMul_1/ReadVariableOpReadVariableOp'gru_28_matmul_1_readvariableop_resource*
_output_shapes
:	0�*
dtype02 
gru_28/MatMul_1/ReadVariableOp�
gru_28/MatMul_1MatMulgru_28/zeros:output:0&gru_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_28/MatMul_1�
gru_28/BiasAdd_1BiasAddgru_28/MatMul_1:product:0gru_28/unstack:output:1*
T0*(
_output_shapes
:����������2
gru_28/BiasAdd_1u
gru_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2
gru_28/Const_1
gru_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_28/split_1/split_dim�
gru_28/split_1SplitVgru_28/BiasAdd_1:output:0gru_28/Const_1:output:0!gru_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
gru_28/split_1�

gru_28/addAddV2gru_28/split:output:0gru_28/split_1:output:0*
T0*'
_output_shapes
:���������02

gru_28/addm
gru_28/SigmoidSigmoidgru_28/add:z:0*
T0*'
_output_shapes
:���������02
gru_28/Sigmoid�
gru_28/add_1AddV2gru_28/split:output:1gru_28/split_1:output:1*
T0*'
_output_shapes
:���������02
gru_28/add_1s
gru_28/Sigmoid_1Sigmoidgru_28/add_1:z:0*
T0*'
_output_shapes
:���������02
gru_28/Sigmoid_1�

gru_28/mulMulgru_28/Sigmoid_1:y:0gru_28/split_1:output:2*
T0*'
_output_shapes
:���������02

gru_28/mul~
gru_28/add_2AddV2gru_28/split:output:2gru_28/mul:z:0*
T0*'
_output_shapes
:���������02
gru_28/add_2f
gru_28/TanhTanhgru_28/add_2:z:0*
T0*'
_output_shapes
:���������02
gru_28/Tanh�
gru_28/mul_1Mulgru_28/Sigmoid:y:0gru_28/zeros:output:0*
T0*'
_output_shapes
:���������02
gru_28/mul_1a
gru_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_28/sub/x|

gru_28/subSubgru_28/sub/x:output:0gru_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������02

gru_28/subv
gru_28/mul_2Mulgru_28/sub:z:0gru_28/Tanh:y:0*
T0*'
_output_shapes
:���������02
gru_28/mul_2{
gru_28/add_3AddV2gru_28/mul_1:z:0gru_28/mul_2:z:0*
T0*'
_output_shapes
:���������02
gru_28/add_3�
$gru_28/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   2&
$gru_28/TensorArrayV2_1/element_shape�
gru_28/TensorArrayV2_1TensorListReserve-gru_28/TensorArrayV2_1/element_shape:output:0gru_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_28/TensorArrayV2_1\
gru_28/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_28/time�
gru_28/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
gru_28/while/maximum_iterationsx
gru_28/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_28/while/loop_counter�
gru_28/whileWhile"gru_28/while/loop_counter:output:0(gru_28/while/maximum_iterations:output:0gru_28/time:output:0gru_28/TensorArrayV2_1:handle:0gru_28/zeros:output:0gru_28/strided_slice_1:output:0>gru_28/TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_28_readvariableop_resource%gru_28_matmul_readvariableop_resource'gru_28_matmul_1_readvariableop_resource^gru_28/MatMul/ReadVariableOp^gru_28/MatMul_1/ReadVariableOp^gru_28/ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������0: : : : : *"
bodyR
gru_28_while_body_2846*"
condR
gru_28_while_cond_2845*8
output_shapes'
%: : : : :���������0: : : : : *
parallel_iterations 2
gru_28/while�
7gru_28/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   29
7gru_28/TensorArrayV2Stack/TensorListStack/element_shape�
)gru_28/TensorArrayV2Stack/TensorListStackTensorListStackgru_28/while:output:3@gru_28/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������0*
element_dtype02+
)gru_28/TensorArrayV2Stack/TensorListStack�
gru_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
gru_28/strided_slice_3/stack�
gru_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_28/strided_slice_3/stack_1�
gru_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_3/stack_2�
gru_28/strided_slice_3StridedSlice2gru_28/TensorArrayV2Stack/TensorListStack:tensor:0%gru_28/strided_slice_3/stack:output:0'gru_28/strided_slice_3/stack_1:output:0'gru_28/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������0*
shrink_axis_mask2
gru_28/strided_slice_3�
gru_28/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_28/transpose_1/perm�
gru_28/transpose_1	Transpose2gru_28/TensorArrayV2Stack/TensorListStack:tensor:0 gru_28/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������02
gru_28/transpose_1�
dense_84/MatMul/ReadVariableOpReadVariableOp'dense_84_matmul_readvariableop_resource*
_output_shapes
:	0�*
dtype02 
dense_84/MatMul/ReadVariableOp�
dense_84/MatMulMatMulgru_28/strided_slice_3:output:0&dense_84/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_84/MatMul�
dense_84/BiasAdd/ReadVariableOpReadVariableOp(dense_84_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_84/BiasAdd/ReadVariableOp�
dense_84/BiasAddBiasAdddense_84/MatMul:product:0'dense_84/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_84/BiasAddt
dense_84/ReluReludense_84/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_84/Relu�
dense_85/MatMul/ReadVariableOpReadVariableOp'dense_85_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_85/MatMul/ReadVariableOp�
dense_85/MatMulMatMuldense_84/Relu:activations:0&dense_85/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_85/MatMul�
dense_85/BiasAdd/ReadVariableOpReadVariableOp(dense_85_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_85/BiasAdd/ReadVariableOp�
dense_85/BiasAddBiasAdddense_85/MatMul:product:0'dense_85/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_85/BiasAddt
dense_85/ReluReludense_85/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_85/Relu�
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_86/MatMul/ReadVariableOp�
dense_86/MatMulMatMuldense_85/Relu:activations:0&dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_86/MatMul�
dense_86/BiasAdd/ReadVariableOpReadVariableOp(dense_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_86/BiasAdd/ReadVariableOp�
dense_86/BiasAddBiasAdddense_86/MatMul:product:0'dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_86/BiasAdd�
IdentityIdentitydense_86/BiasAdd:output:0 ^dense_84/BiasAdd/ReadVariableOp^dense_84/MatMul/ReadVariableOp ^dense_85/BiasAdd/ReadVariableOp^dense_85/MatMul/ReadVariableOp ^dense_86/BiasAdd/ReadVariableOp^dense_86/MatMul/ReadVariableOp^gru_28/MatMul/ReadVariableOp^gru_28/MatMul_1/ReadVariableOp^gru_28/ReadVariableOp^gru_28/while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������:::::::::2B
dense_84/BiasAdd/ReadVariableOpdense_84/BiasAdd/ReadVariableOp2@
dense_84/MatMul/ReadVariableOpdense_84/MatMul/ReadVariableOp2B
dense_85/BiasAdd/ReadVariableOpdense_85/BiasAdd/ReadVariableOp2@
dense_85/MatMul/ReadVariableOpdense_85/MatMul/ReadVariableOp2B
dense_86/BiasAdd/ReadVariableOpdense_86/BiasAdd/ReadVariableOp2@
dense_86/MatMul/ReadVariableOpdense_86/MatMul/ReadVariableOp2<
gru_28/MatMul/ReadVariableOpgru_28/MatMul/ReadVariableOp2@
gru_28/MatMul_1/ReadVariableOpgru_28/MatMul_1/ReadVariableOp2.
gru_28/ReadVariableOpgru_28/ReadVariableOp2
gru_28/whilegru_28/while:& "
 
_user_specified_nameinputs
�
�
while_cond_1912
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1,
(while_cond_1912___redundant_placeholder0,
(while_cond_1912___redundant_placeholder1,
(while_cond_1912___redundant_placeholder2,
(while_cond_1912___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :���������0: ::::
�2
�
gru_28_while_body_2846
gru_28_while_loop_counter#
gru_28_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
gru_28_strided_slice_1_0X
Ttensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
gru_28_strided_slice_1V
Rtensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemTtensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem{
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/ye
add_5AddV2gru_28_while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5�
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitygru_28_while_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������02

Identity_4"2
gru_28_strided_slice_1gru_28_strided_slice_1_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0"�
Rtensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensorTtensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������0: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
�
�
while_cond_3383
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1,
(while_cond_3383___redundant_placeholder0,
(while_cond_3383___redundant_placeholder1,
(while_cond_3383___redundant_placeholder2,
(while_cond_3383___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :���������0: ::::
�	
�
'__inference_gru_cell_layer_call_fn_3791

inputs
states_0"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*:
_output_shapes(
&:���������0:���������0**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_16372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������02

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������02

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:���������0:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
�1
�
while_body_3384
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem{
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5�
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������0: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
�
�
B__inference_gru_cell_layer_call_and_return_conditional_losses_3780

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanh^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������02

Identity�

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������02

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:���������0:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
�
�
while_body_1913
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 statefulpartitionedcall_args_2_0$
 statefulpartitionedcall_args_3_0$
 statefulpartitionedcall_args_4_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4��StatefulPartitionedCall�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2 statefulpartitionedcall_args_2_0 statefulpartitionedcall_args_3_0 statefulpartitionedcall_args_4_0*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*:
_output_shapes(
&:���������0:���������0**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_16372
StatefulPartitionedCall�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1f
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identityy

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1h

Identity_2Identityadd:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"B
statefulpartitionedcall_args_2 statefulpartitionedcall_args_2_0"B
statefulpartitionedcall_args_3 statefulpartitionedcall_args_3_0"B
statefulpartitionedcall_args_4 statefulpartitionedcall_args_4_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������0: : :::22
StatefulPartitionedCallStatefulPartitionedCall
�
�
%__inference_gru_28_layer_call_fn_3307
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������0**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_gru_28_layer_call_and_return_conditional_losses_19732
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������02

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
�
�
while_body_2021
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 statefulpartitionedcall_args_2_0$
 statefulpartitionedcall_args_3_0$
 statefulpartitionedcall_args_4_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4��StatefulPartitionedCall�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem�
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2 statefulpartitionedcall_args_2_0 statefulpartitionedcall_args_3_0 statefulpartitionedcall_args_4_0*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*:
_output_shapes(
&:���������0:���������0**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_16772
StatefulPartitionedCall�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1f
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identityy

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1h

Identity_2Identityadd:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"B
statefulpartitionedcall_args_2 statefulpartitionedcall_args_2_0"B
statefulpartitionedcall_args_3 statefulpartitionedcall_args_3_0"B
statefulpartitionedcall_args_4 statefulpartitionedcall_args_4_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������0: : :::22
StatefulPartitionedCallStatefulPartitionedCall
��
�
 __inference__traced_restore_4030
file_prefix$
 assignvariableop_dense_84_kernel$
 assignvariableop_1_dense_84_bias&
"assignvariableop_2_dense_85_kernel$
 assignvariableop_3_dense_85_bias&
"assignvariableop_4_dense_86_kernel$
 assignvariableop_5_dense_86_bias)
%assignvariableop_6_training_adam_iter+
'assignvariableop_7_training_adam_beta_1+
'assignvariableop_8_training_adam_beta_2*
&assignvariableop_9_training_adam_decay3
/assignvariableop_10_training_adam_learning_rate%
!assignvariableop_11_gru_28_kernel/
+assignvariableop_12_gru_28_recurrent_kernel#
assignvariableop_13_gru_28_bias7
3assignvariableop_14_training_adam_dense_84_kernel_m5
1assignvariableop_15_training_adam_dense_84_bias_m7
3assignvariableop_16_training_adam_dense_85_kernel_m5
1assignvariableop_17_training_adam_dense_85_bias_m7
3assignvariableop_18_training_adam_dense_86_kernel_m5
1assignvariableop_19_training_adam_dense_86_bias_m5
1assignvariableop_20_training_adam_gru_28_kernel_m?
;assignvariableop_21_training_adam_gru_28_recurrent_kernel_m3
/assignvariableop_22_training_adam_gru_28_bias_m7
3assignvariableop_23_training_adam_dense_84_kernel_v5
1assignvariableop_24_training_adam_dense_84_bias_v7
3assignvariableop_25_training_adam_dense_85_kernel_v5
1assignvariableop_26_training_adam_dense_85_bias_v7
3assignvariableop_27_training_adam_dense_86_kernel_v5
1assignvariableop_28_training_adam_dense_86_bias_v5
1assignvariableop_29_training_adam_gru_28_kernel_v?
;assignvariableop_30_training_adam_gru_28_recurrent_kernel_v3
/assignvariableop_31_training_adam_gru_28_bias_v
identity_33��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_dense_84_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_84_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_85_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_85_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_86_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_86_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0	*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_training_adam_iterIdentity_6:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp'assignvariableop_7_training_adam_beta_1Identity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp'assignvariableop_8_training_adam_beta_2Identity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp&assignvariableop_9_training_adam_decayIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_training_adam_learning_rateIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_gru_28_kernelIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp+assignvariableop_12_gru_28_recurrent_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_gru_28_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp3assignvariableop_14_training_adam_dense_84_kernel_mIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp1assignvariableop_15_training_adam_dense_84_bias_mIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp3assignvariableop_16_training_adam_dense_85_kernel_mIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp1assignvariableop_17_training_adam_dense_85_bias_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp3assignvariableop_18_training_adam_dense_86_kernel_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp1assignvariableop_19_training_adam_dense_86_bias_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp1assignvariableop_20_training_adam_gru_28_kernel_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp;assignvariableop_21_training_adam_gru_28_recurrent_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp/assignvariableop_22_training_adam_gru_28_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp3assignvariableop_23_training_adam_dense_84_kernel_vIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp1assignvariableop_24_training_adam_dense_84_bias_vIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp3assignvariableop_25_training_adam_dense_85_kernel_vIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp1assignvariableop_26_training_adam_dense_85_bias_vIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp3assignvariableop_27_training_adam_dense_86_kernel_vIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp1assignvariableop_28_training_adam_dense_86_bias_vIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp1assignvariableop_29_training_adam_gru_28_kernel_vIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp;assignvariableop_30_training_adam_gru_28_recurrent_kernel_vIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp/assignvariableop_31_training_adam_gru_28_bias_vIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_32Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_32�
Identity_33IdentityIdentity_32:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_33"#
identity_33Identity_33:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
�

�
"__inference_signature_wrapper_2599
gru_28_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgru_28_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9*
Tin
2
*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__wrapped_model_15652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:, (
&
_user_specified_namegru_28_input
�	
�
B__inference_dense_85_layer_call_and_return_conditional_losses_2461

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
G__inference_sequential_28_layer_call_and_return_conditional_losses_2513
gru_28_input)
%gru_28_statefulpartitionedcall_args_1)
%gru_28_statefulpartitionedcall_args_2)
%gru_28_statefulpartitionedcall_args_3+
'dense_84_statefulpartitionedcall_args_1+
'dense_84_statefulpartitionedcall_args_2+
'dense_85_statefulpartitionedcall_args_1+
'dense_85_statefulpartitionedcall_args_2+
'dense_86_statefulpartitionedcall_args_1+
'dense_86_statefulpartitionedcall_args_2
identity�� dense_84/StatefulPartitionedCall� dense_85/StatefulPartitionedCall� dense_86/StatefulPartitionedCall�gru_28/StatefulPartitionedCall�
gru_28/StatefulPartitionedCallStatefulPartitionedCallgru_28_input%gru_28_statefulpartitionedcall_args_1%gru_28_statefulpartitionedcall_args_2%gru_28_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������0**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_gru_28_layer_call_and_return_conditional_losses_24062 
gru_28/StatefulPartitionedCall�
 dense_84/StatefulPartitionedCallStatefulPartitionedCall'gru_28/StatefulPartitionedCall:output:0'dense_84_statefulpartitionedcall_args_1'dense_84_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_84_layer_call_and_return_conditional_losses_24382"
 dense_84/StatefulPartitionedCall�
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0'dense_85_statefulpartitionedcall_args_1'dense_85_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_85_layer_call_and_return_conditional_losses_24612"
 dense_85/StatefulPartitionedCall�
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0'dense_86_statefulpartitionedcall_args_1'dense_86_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_86_layer_call_and_return_conditional_losses_24832"
 dense_86/StatefulPartitionedCall�
IdentityIdentity)dense_86/StatefulPartitionedCall:output:0!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall^gru_28/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������:::::::::2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2@
gru_28/StatefulPartitionedCallgru_28/StatefulPartitionedCall:, (
&
_user_specified_namegru_28_input
�	
�
B__inference_dense_84_layer_call_and_return_conditional_losses_2438

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	0�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������0::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
G__inference_sequential_28_layer_call_and_return_conditional_losses_2533

inputs)
%gru_28_statefulpartitionedcall_args_1)
%gru_28_statefulpartitionedcall_args_2)
%gru_28_statefulpartitionedcall_args_3+
'dense_84_statefulpartitionedcall_args_1+
'dense_84_statefulpartitionedcall_args_2+
'dense_85_statefulpartitionedcall_args_1+
'dense_85_statefulpartitionedcall_args_2+
'dense_86_statefulpartitionedcall_args_1+
'dense_86_statefulpartitionedcall_args_2
identity�� dense_84/StatefulPartitionedCall� dense_85/StatefulPartitionedCall� dense_86/StatefulPartitionedCall�gru_28/StatefulPartitionedCall�
gru_28/StatefulPartitionedCallStatefulPartitionedCallinputs%gru_28_statefulpartitionedcall_args_1%gru_28_statefulpartitionedcall_args_2%gru_28_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������0**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_gru_28_layer_call_and_return_conditional_losses_22482 
gru_28/StatefulPartitionedCall�
 dense_84/StatefulPartitionedCallStatefulPartitionedCall'gru_28/StatefulPartitionedCall:output:0'dense_84_statefulpartitionedcall_args_1'dense_84_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_84_layer_call_and_return_conditional_losses_24382"
 dense_84/StatefulPartitionedCall�
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0'dense_85_statefulpartitionedcall_args_1'dense_85_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_85_layer_call_and_return_conditional_losses_24612"
 dense_85/StatefulPartitionedCall�
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0'dense_86_statefulpartitionedcall_args_1'dense_86_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_86_layer_call_and_return_conditional_losses_24832"
 dense_86/StatefulPartitionedCall�
IdentityIdentity)dense_86/StatefulPartitionedCall:output:0!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall^gru_28/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������:::::::::2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2@
gru_28/StatefulPartitionedCallgru_28/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
%__inference_gru_28_layer_call_fn_3647

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������0**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_gru_28_layer_call_and_return_conditional_losses_24062
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�P
�
@__inference_gru_28_layer_call_and_return_conditional_losses_2406

inputs
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :02
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :02
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������02
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanhd
mul_1MulSigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcematmul_readvariableop_resource matmul_1_readvariableop_resource^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������0: : : : : *
bodyR
while_body_2317*
condR
while_cond_2316*8
output_shapes'
%: : : : :���������0: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������0*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������0*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������02
transpose_1�
IdentityIdentitystrided_slice_3:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:���������02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs
�
�
,__inference_sequential_28_layer_call_fn_2576
gru_28_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgru_28_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9*
Tin
2
*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_28_layer_call_and_return_conditional_losses_25642
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:, (
&
_user_specified_namegru_28_input
�x
�
G__inference_sequential_28_layer_call_and_return_conditional_losses_2777

inputs"
gru_28_readvariableop_resource)
%gru_28_matmul_readvariableop_resource+
'gru_28_matmul_1_readvariableop_resource+
'dense_84_matmul_readvariableop_resource,
(dense_84_biasadd_readvariableop_resource+
'dense_85_matmul_readvariableop_resource,
(dense_85_biasadd_readvariableop_resource+
'dense_86_matmul_readvariableop_resource,
(dense_86_biasadd_readvariableop_resource
identity��dense_84/BiasAdd/ReadVariableOp�dense_84/MatMul/ReadVariableOp�dense_85/BiasAdd/ReadVariableOp�dense_85/MatMul/ReadVariableOp�dense_86/BiasAdd/ReadVariableOp�dense_86/MatMul/ReadVariableOp�gru_28/MatMul/ReadVariableOp�gru_28/MatMul_1/ReadVariableOp�gru_28/ReadVariableOp�gru_28/whileR
gru_28/ShapeShapeinputs*
T0*
_output_shapes
:2
gru_28/Shape�
gru_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_28/strided_slice/stack�
gru_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru_28/strided_slice/stack_1�
gru_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru_28/strided_slice/stack_2�
gru_28/strided_sliceStridedSlicegru_28/Shape:output:0#gru_28/strided_slice/stack:output:0%gru_28/strided_slice/stack_1:output:0%gru_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_28/strided_slicej
gru_28/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :02
gru_28/zeros/mul/y�
gru_28/zeros/mulMulgru_28/strided_slice:output:0gru_28/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru_28/zeros/mulm
gru_28/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
gru_28/zeros/Less/y�
gru_28/zeros/LessLessgru_28/zeros/mul:z:0gru_28/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru_28/zeros/Lessp
gru_28/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :02
gru_28/zeros/packed/1�
gru_28/zeros/packedPackgru_28/strided_slice:output:0gru_28/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru_28/zeros/packedm
gru_28/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru_28/zeros/Const�
gru_28/zerosFillgru_28/zeros/packed:output:0gru_28/zeros/Const:output:0*
T0*'
_output_shapes
:���������02
gru_28/zeros�
gru_28/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_28/transpose/perm�
gru_28/transpose	Transposeinputsgru_28/transpose/perm:output:0*
T0*+
_output_shapes
:���������2
gru_28/transposed
gru_28/Shape_1Shapegru_28/transpose:y:0*
T0*
_output_shapes
:2
gru_28/Shape_1�
gru_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_28/strided_slice_1/stack�
gru_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_1/stack_1�
gru_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_1/stack_2�
gru_28/strided_slice_1StridedSlicegru_28/Shape_1:output:0%gru_28/strided_slice_1/stack:output:0'gru_28/strided_slice_1/stack_1:output:0'gru_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru_28/strided_slice_1�
"gru_28/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2$
"gru_28/TensorArrayV2/element_shape�
gru_28/TensorArrayV2TensorListReserve+gru_28/TensorArrayV2/element_shape:output:0gru_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_28/TensorArrayV2�
<gru_28/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2>
<gru_28/TensorArrayUnstack/TensorListFromTensor/element_shape�
.gru_28/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_28/transpose:y:0Egru_28/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.gru_28/TensorArrayUnstack/TensorListFromTensor�
gru_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru_28/strided_slice_2/stack�
gru_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_2/stack_1�
gru_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_2/stack_2�
gru_28/strided_slice_2StridedSlicegru_28/transpose:y:0%gru_28/strided_slice_2/stack:output:0'gru_28/strided_slice_2/stack_1:output:0'gru_28/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
gru_28/strided_slice_2�
gru_28/ReadVariableOpReadVariableOpgru_28_readvariableop_resource*
_output_shapes
:	�*
dtype02
gru_28/ReadVariableOp�
gru_28/unstackUnpackgru_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
gru_28/unstack�
gru_28/MatMul/ReadVariableOpReadVariableOp%gru_28_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
gru_28/MatMul/ReadVariableOp�
gru_28/MatMulMatMulgru_28/strided_slice_2:output:0$gru_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_28/MatMul�
gru_28/BiasAddBiasAddgru_28/MatMul:product:0gru_28/unstack:output:0*
T0*(
_output_shapes
:����������2
gru_28/BiasAdd^
gru_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_28/Const{
gru_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_28/split/split_dim�
gru_28/splitSplitgru_28/split/split_dim:output:0gru_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
gru_28/split�
gru_28/MatMul_1/ReadVariableOpReadVariableOp'gru_28_matmul_1_readvariableop_resource*
_output_shapes
:	0�*
dtype02 
gru_28/MatMul_1/ReadVariableOp�
gru_28/MatMul_1MatMulgru_28/zeros:output:0&gru_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
gru_28/MatMul_1�
gru_28/BiasAdd_1BiasAddgru_28/MatMul_1:product:0gru_28/unstack:output:1*
T0*(
_output_shapes
:����������2
gru_28/BiasAdd_1u
gru_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2
gru_28/Const_1
gru_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
gru_28/split_1/split_dim�
gru_28/split_1SplitVgru_28/BiasAdd_1:output:0gru_28/Const_1:output:0!gru_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
gru_28/split_1�

gru_28/addAddV2gru_28/split:output:0gru_28/split_1:output:0*
T0*'
_output_shapes
:���������02

gru_28/addm
gru_28/SigmoidSigmoidgru_28/add:z:0*
T0*'
_output_shapes
:���������02
gru_28/Sigmoid�
gru_28/add_1AddV2gru_28/split:output:1gru_28/split_1:output:1*
T0*'
_output_shapes
:���������02
gru_28/add_1s
gru_28/Sigmoid_1Sigmoidgru_28/add_1:z:0*
T0*'
_output_shapes
:���������02
gru_28/Sigmoid_1�

gru_28/mulMulgru_28/Sigmoid_1:y:0gru_28/split_1:output:2*
T0*'
_output_shapes
:���������02

gru_28/mul~
gru_28/add_2AddV2gru_28/split:output:2gru_28/mul:z:0*
T0*'
_output_shapes
:���������02
gru_28/add_2f
gru_28/TanhTanhgru_28/add_2:z:0*
T0*'
_output_shapes
:���������02
gru_28/Tanh�
gru_28/mul_1Mulgru_28/Sigmoid:y:0gru_28/zeros:output:0*
T0*'
_output_shapes
:���������02
gru_28/mul_1a
gru_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
gru_28/sub/x|

gru_28/subSubgru_28/sub/x:output:0gru_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������02

gru_28/subv
gru_28/mul_2Mulgru_28/sub:z:0gru_28/Tanh:y:0*
T0*'
_output_shapes
:���������02
gru_28/mul_2{
gru_28/add_3AddV2gru_28/mul_1:z:0gru_28/mul_2:z:0*
T0*'
_output_shapes
:���������02
gru_28/add_3�
$gru_28/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   2&
$gru_28/TensorArrayV2_1/element_shape�
gru_28/TensorArrayV2_1TensorListReserve-gru_28/TensorArrayV2_1/element_shape:output:0gru_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru_28/TensorArrayV2_1\
gru_28/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_28/time�
gru_28/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2!
gru_28/while/maximum_iterationsx
gru_28/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru_28/while/loop_counter�
gru_28/whileWhile"gru_28/while/loop_counter:output:0(gru_28/while/maximum_iterations:output:0gru_28/time:output:0gru_28/TensorArrayV2_1:handle:0gru_28/zeros:output:0gru_28/strided_slice_1:output:0>gru_28/TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_28_readvariableop_resource%gru_28_matmul_readvariableop_resource'gru_28_matmul_1_readvariableop_resource^gru_28/MatMul/ReadVariableOp^gru_28/MatMul_1/ReadVariableOp^gru_28/ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������0: : : : : *"
bodyR
gru_28_while_body_2668*"
condR
gru_28_while_cond_2667*8
output_shapes'
%: : : : :���������0: : : : : *
parallel_iterations 2
gru_28/while�
7gru_28/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   29
7gru_28/TensorArrayV2Stack/TensorListStack/element_shape�
)gru_28/TensorArrayV2Stack/TensorListStackTensorListStackgru_28/while:output:3@gru_28/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������0*
element_dtype02+
)gru_28/TensorArrayV2Stack/TensorListStack�
gru_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
gru_28/strided_slice_3/stack�
gru_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
gru_28/strided_slice_3/stack_1�
gru_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru_28/strided_slice_3/stack_2�
gru_28/strided_slice_3StridedSlice2gru_28/TensorArrayV2Stack/TensorListStack:tensor:0%gru_28/strided_slice_3/stack:output:0'gru_28/strided_slice_3/stack_1:output:0'gru_28/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������0*
shrink_axis_mask2
gru_28/strided_slice_3�
gru_28/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru_28/transpose_1/perm�
gru_28/transpose_1	Transpose2gru_28/TensorArrayV2Stack/TensorListStack:tensor:0 gru_28/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������02
gru_28/transpose_1�
dense_84/MatMul/ReadVariableOpReadVariableOp'dense_84_matmul_readvariableop_resource*
_output_shapes
:	0�*
dtype02 
dense_84/MatMul/ReadVariableOp�
dense_84/MatMulMatMulgru_28/strided_slice_3:output:0&dense_84/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_84/MatMul�
dense_84/BiasAdd/ReadVariableOpReadVariableOp(dense_84_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_84/BiasAdd/ReadVariableOp�
dense_84/BiasAddBiasAdddense_84/MatMul:product:0'dense_84/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_84/BiasAddt
dense_84/ReluReludense_84/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_84/Relu�
dense_85/MatMul/ReadVariableOpReadVariableOp'dense_85_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02 
dense_85/MatMul/ReadVariableOp�
dense_85/MatMulMatMuldense_84/Relu:activations:0&dense_85/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_85/MatMul�
dense_85/BiasAdd/ReadVariableOpReadVariableOp(dense_85_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
dense_85/BiasAdd/ReadVariableOp�
dense_85/BiasAddBiasAdddense_85/MatMul:product:0'dense_85/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_85/BiasAddt
dense_85/ReluReludense_85/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_85/Relu�
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_86/MatMul/ReadVariableOp�
dense_86/MatMulMatMuldense_85/Relu:activations:0&dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_86/MatMul�
dense_86/BiasAdd/ReadVariableOpReadVariableOp(dense_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_86/BiasAdd/ReadVariableOp�
dense_86/BiasAddBiasAdddense_86/MatMul:product:0'dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_86/BiasAdd�
IdentityIdentitydense_86/BiasAdd:output:0 ^dense_84/BiasAdd/ReadVariableOp^dense_84/MatMul/ReadVariableOp ^dense_85/BiasAdd/ReadVariableOp^dense_85/MatMul/ReadVariableOp ^dense_86/BiasAdd/ReadVariableOp^dense_86/MatMul/ReadVariableOp^gru_28/MatMul/ReadVariableOp^gru_28/MatMul_1/ReadVariableOp^gru_28/ReadVariableOp^gru_28/while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������:::::::::2B
dense_84/BiasAdd/ReadVariableOpdense_84/BiasAdd/ReadVariableOp2@
dense_84/MatMul/ReadVariableOpdense_84/MatMul/ReadVariableOp2B
dense_85/BiasAdd/ReadVariableOpdense_85/BiasAdd/ReadVariableOp2@
dense_85/MatMul/ReadVariableOpdense_85/MatMul/ReadVariableOp2B
dense_86/BiasAdd/ReadVariableOpdense_86/BiasAdd/ReadVariableOp2@
dense_86/MatMul/ReadVariableOpdense_86/MatMul/ReadVariableOp2<
gru_28/MatMul/ReadVariableOpgru_28/MatMul/ReadVariableOp2@
gru_28/MatMul_1/ReadVariableOpgru_28/MatMul_1/ReadVariableOp2.
gru_28/ReadVariableOpgru_28/ReadVariableOp2
gru_28/whilegru_28/while:& "
 
_user_specified_nameinputs
�	
�
B__inference_dense_84_layer_call_and_return_conditional_losses_3658

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	0�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������0::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�;
�
@__inference_gru_28_layer_call_and_return_conditional_losses_2081

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :02
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :02
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������02
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*:
_output_shapes(
&:���������0:���������0**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_16772
StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4^StatefulPartitionedCall*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������0: : : : : *
bodyR
while_body_2021*
condR
while_cond_2020*8
output_shapes'
%: : : : :���������0: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������0*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������0*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������02
transpose_1�
IdentityIdentitystrided_slice_3:output:0^StatefulPartitionedCall^while*
T0*'
_output_shapes
:���������02

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:& "
 
_user_specified_nameinputs
�
�
B__inference_dense_86_layer_call_and_return_conditional_losses_3693

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�2
�
gru_28_while_body_2668
gru_28_while_loop_counter#
gru_28_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
gru_28_strided_slice_1_0X
Ttensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
gru_28_strided_slice_1V
Rtensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemTtensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem{
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/ye
add_5AddV2gru_28_while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5�
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitygru_28_while_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������02

Identity_4"2
gru_28_strided_slice_1gru_28_strided_slice_1_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0"�
Rtensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensorTtensorarrayv2read_tensorlistgetitem_gru_28_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������0: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
�
�
'__inference_dense_85_layer_call_fn_3683

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_85_layer_call_and_return_conditional_losses_24612
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
'__inference_dense_84_layer_call_fn_3665

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_84_layer_call_and_return_conditional_losses_24382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������0::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
$sequential_28_gru_28_while_cond_1455+
'sequential_28_gru_28_while_loop_counter1
-sequential_28_gru_28_while_maximum_iterations
placeholder
placeholder_1
placeholder_2-
)less_sequential_28_gru_28_strided_slice_1A
=sequential_28_gru_28_while_cond_1455___redundant_placeholder0A
=sequential_28_gru_28_while_cond_1455___redundant_placeholder1A
=sequential_28_gru_28_while_cond_1455___redundant_placeholder2A
=sequential_28_gru_28_while_cond_1455___redundant_placeholder3
identity
m
LessLessplaceholder)less_sequential_28_gru_28_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :���������0: ::::
�
�
B__inference_gru_cell_layer_call_and_return_conditional_losses_3740

inputs
states_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity

identity_1��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOpy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanh^
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
IdentityIdentity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������02

Identity�

Identity_1Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������02

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:���������0:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
�P
�
@__inference_gru_28_layer_call_and_return_conditional_losses_2248

inputs
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :02
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :02
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������02
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanhd
mul_1MulSigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcematmul_readvariableop_resource matmul_1_readvariableop_resource^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������0: : : : : *
bodyR
while_body_2159*
condR
while_cond_2158*8
output_shapes'
%: : : : :���������0: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������0*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������0*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������02
transpose_1�
IdentityIdentitystrided_slice_3:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:���������02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs
�P
�
@__inference_gru_28_layer_call_and_return_conditional_losses_3631

inputs
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :02
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :02
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������02
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanhd
mul_1MulSigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcematmul_readvariableop_resource matmul_1_readvariableop_resource^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������0: : : : : *
bodyR
while_body_3542*
condR
while_cond_3541*8
output_shapes'
%: : : : :���������0: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������0*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������0*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������02
transpose_1�
IdentityIdentitystrided_slice_3:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:���������02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs
�P
�
@__inference_gru_28_layer_call_and_return_conditional_losses_3299
inputs_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :02
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :02
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������02
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanhd
mul_1MulSigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcematmul_readvariableop_resource matmul_1_readvariableop_resource^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������0: : : : : *
bodyR
while_body_3210*
condR
while_cond_3209*8
output_shapes'
%: : : : :���������0: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������0*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������0*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������02
transpose_1�
IdentityIdentitystrided_slice_3:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:���������02

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:( $
"
_user_specified_name
inputs/0
�	
�
'__inference_gru_cell_layer_call_fn_3802

inputs
states_0"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*:
_output_shapes(
&:���������0:���������0**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_16772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������02

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:���������02

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:���������:���������0:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
�
�
while_cond_3051
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1,
(while_cond_3051___redundant_placeholder0,
(while_cond_3051___redundant_placeholder1,
(while_cond_3051___redundant_placeholder2,
(while_cond_3051___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :���������0: ::::
�1
�
while_body_3052
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem{
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5�
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������0: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
�E
�
__inference__traced_save_3922
file_prefix.
*savev2_dense_84_kernel_read_readvariableop,
(savev2_dense_84_bias_read_readvariableop.
*savev2_dense_85_kernel_read_readvariableop,
(savev2_dense_85_bias_read_readvariableop.
*savev2_dense_86_kernel_read_readvariableop,
(savev2_dense_86_bias_read_readvariableop1
-savev2_training_adam_iter_read_readvariableop	3
/savev2_training_adam_beta_1_read_readvariableop3
/savev2_training_adam_beta_2_read_readvariableop2
.savev2_training_adam_decay_read_readvariableop:
6savev2_training_adam_learning_rate_read_readvariableop,
(savev2_gru_28_kernel_read_readvariableop6
2savev2_gru_28_recurrent_kernel_read_readvariableop*
&savev2_gru_28_bias_read_readvariableop>
:savev2_training_adam_dense_84_kernel_m_read_readvariableop<
8savev2_training_adam_dense_84_bias_m_read_readvariableop>
:savev2_training_adam_dense_85_kernel_m_read_readvariableop<
8savev2_training_adam_dense_85_bias_m_read_readvariableop>
:savev2_training_adam_dense_86_kernel_m_read_readvariableop<
8savev2_training_adam_dense_86_bias_m_read_readvariableop<
8savev2_training_adam_gru_28_kernel_m_read_readvariableopF
Bsavev2_training_adam_gru_28_recurrent_kernel_m_read_readvariableop:
6savev2_training_adam_gru_28_bias_m_read_readvariableop>
:savev2_training_adam_dense_84_kernel_v_read_readvariableop<
8savev2_training_adam_dense_84_bias_v_read_readvariableop>
:savev2_training_adam_dense_85_kernel_v_read_readvariableop<
8savev2_training_adam_dense_85_bias_v_read_readvariableop>
:savev2_training_adam_dense_86_kernel_v_read_readvariableop<
8savev2_training_adam_dense_86_bias_v_read_readvariableop<
8savev2_training_adam_gru_28_kernel_v_read_readvariableopF
Bsavev2_training_adam_gru_28_recurrent_kernel_v_read_readvariableop:
6savev2_training_adam_gru_28_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_5c9c89b0d6244000b5b34e8355ad1044/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_84_kernel_read_readvariableop(savev2_dense_84_bias_read_readvariableop*savev2_dense_85_kernel_read_readvariableop(savev2_dense_85_bias_read_readvariableop*savev2_dense_86_kernel_read_readvariableop(savev2_dense_86_bias_read_readvariableop-savev2_training_adam_iter_read_readvariableop/savev2_training_adam_beta_1_read_readvariableop/savev2_training_adam_beta_2_read_readvariableop.savev2_training_adam_decay_read_readvariableop6savev2_training_adam_learning_rate_read_readvariableop(savev2_gru_28_kernel_read_readvariableop2savev2_gru_28_recurrent_kernel_read_readvariableop&savev2_gru_28_bias_read_readvariableop:savev2_training_adam_dense_84_kernel_m_read_readvariableop8savev2_training_adam_dense_84_bias_m_read_readvariableop:savev2_training_adam_dense_85_kernel_m_read_readvariableop8savev2_training_adam_dense_85_bias_m_read_readvariableop:savev2_training_adam_dense_86_kernel_m_read_readvariableop8savev2_training_adam_dense_86_bias_m_read_readvariableop8savev2_training_adam_gru_28_kernel_m_read_readvariableopBsavev2_training_adam_gru_28_recurrent_kernel_m_read_readvariableop6savev2_training_adam_gru_28_bias_m_read_readvariableop:savev2_training_adam_dense_84_kernel_v_read_readvariableop8savev2_training_adam_dense_84_bias_v_read_readvariableop:savev2_training_adam_dense_85_kernel_v_read_readvariableop8savev2_training_adam_dense_85_bias_v_read_readvariableop:savev2_training_adam_dense_86_kernel_v_read_readvariableop8savev2_training_adam_dense_86_bias_v_read_readvariableop8savev2_training_adam_gru_28_kernel_v_read_readvariableopBsavev2_training_adam_gru_28_recurrent_kernel_v_read_readvariableop6savev2_training_adam_gru_28_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	0�:�:
��:�:	�:: : : : : :	�:	0�:	�:	0�:�:
��:�:	�::	�:	0�:	�:	0�:�:
��:�:	�::	�:	0�:	�: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�;
�
@__inference_gru_28_layer_call_and_return_conditional_losses_1973

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :02
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :02
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������02
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*:
_output_shapes(
&:���������0:���������0**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_16372
StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4^StatefulPartitionedCall*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������0: : : : : *
bodyR
while_body_1913*
condR
while_cond_1912*8
output_shapes'
%: : : : :���������0: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������0*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������0*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������02
transpose_1�
IdentityIdentitystrided_slice_3:output:0^StatefulPartitionedCall^while*
T0*'
_output_shapes
:���������02

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:& "
 
_user_specified_nameinputs
�
�
G__inference_sequential_28_layer_call_and_return_conditional_losses_2496
gru_28_input)
%gru_28_statefulpartitionedcall_args_1)
%gru_28_statefulpartitionedcall_args_2)
%gru_28_statefulpartitionedcall_args_3+
'dense_84_statefulpartitionedcall_args_1+
'dense_84_statefulpartitionedcall_args_2+
'dense_85_statefulpartitionedcall_args_1+
'dense_85_statefulpartitionedcall_args_2+
'dense_86_statefulpartitionedcall_args_1+
'dense_86_statefulpartitionedcall_args_2
identity�� dense_84/StatefulPartitionedCall� dense_85/StatefulPartitionedCall� dense_86/StatefulPartitionedCall�gru_28/StatefulPartitionedCall�
gru_28/StatefulPartitionedCallStatefulPartitionedCallgru_28_input%gru_28_statefulpartitionedcall_args_1%gru_28_statefulpartitionedcall_args_2%gru_28_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������0**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_gru_28_layer_call_and_return_conditional_losses_22482 
gru_28/StatefulPartitionedCall�
 dense_84/StatefulPartitionedCallStatefulPartitionedCall'gru_28/StatefulPartitionedCall:output:0'dense_84_statefulpartitionedcall_args_1'dense_84_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_84_layer_call_and_return_conditional_losses_24382"
 dense_84/StatefulPartitionedCall�
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0'dense_85_statefulpartitionedcall_args_1'dense_85_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_85_layer_call_and_return_conditional_losses_24612"
 dense_85/StatefulPartitionedCall�
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0'dense_86_statefulpartitionedcall_args_1'dense_86_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_86_layer_call_and_return_conditional_losses_24832"
 dense_86/StatefulPartitionedCall�
IdentityIdentity)dense_86/StatefulPartitionedCall:output:0!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall^gru_28/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������:::::::::2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2@
gru_28/StatefulPartitionedCallgru_28/StatefulPartitionedCall:, (
&
_user_specified_namegru_28_input
�1
�
while_body_3210
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem{
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5�
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������0: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
��
�
__inference__wrapped_model_1565
gru_28_input0
,sequential_28_gru_28_readvariableop_resource7
3sequential_28_gru_28_matmul_readvariableop_resource9
5sequential_28_gru_28_matmul_1_readvariableop_resource9
5sequential_28_dense_84_matmul_readvariableop_resource:
6sequential_28_dense_84_biasadd_readvariableop_resource9
5sequential_28_dense_85_matmul_readvariableop_resource:
6sequential_28_dense_85_biasadd_readvariableop_resource9
5sequential_28_dense_86_matmul_readvariableop_resource:
6sequential_28_dense_86_biasadd_readvariableop_resource
identity��-sequential_28/dense_84/BiasAdd/ReadVariableOp�,sequential_28/dense_84/MatMul/ReadVariableOp�-sequential_28/dense_85/BiasAdd/ReadVariableOp�,sequential_28/dense_85/MatMul/ReadVariableOp�-sequential_28/dense_86/BiasAdd/ReadVariableOp�,sequential_28/dense_86/MatMul/ReadVariableOp�*sequential_28/gru_28/MatMul/ReadVariableOp�,sequential_28/gru_28/MatMul_1/ReadVariableOp�#sequential_28/gru_28/ReadVariableOp�sequential_28/gru_28/whilet
sequential_28/gru_28/ShapeShapegru_28_input*
T0*
_output_shapes
:2
sequential_28/gru_28/Shape�
(sequential_28/gru_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_28/gru_28/strided_slice/stack�
*sequential_28/gru_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_28/gru_28/strided_slice/stack_1�
*sequential_28/gru_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_28/gru_28/strided_slice/stack_2�
"sequential_28/gru_28/strided_sliceStridedSlice#sequential_28/gru_28/Shape:output:01sequential_28/gru_28/strided_slice/stack:output:03sequential_28/gru_28/strided_slice/stack_1:output:03sequential_28/gru_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_28/gru_28/strided_slice�
 sequential_28/gru_28/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :02"
 sequential_28/gru_28/zeros/mul/y�
sequential_28/gru_28/zeros/mulMul+sequential_28/gru_28/strided_slice:output:0)sequential_28/gru_28/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_28/gru_28/zeros/mul�
!sequential_28/gru_28/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2#
!sequential_28/gru_28/zeros/Less/y�
sequential_28/gru_28/zeros/LessLess"sequential_28/gru_28/zeros/mul:z:0*sequential_28/gru_28/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_28/gru_28/zeros/Less�
#sequential_28/gru_28/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :02%
#sequential_28/gru_28/zeros/packed/1�
!sequential_28/gru_28/zeros/packedPack+sequential_28/gru_28/strided_slice:output:0,sequential_28/gru_28/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_28/gru_28/zeros/packed�
 sequential_28/gru_28/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_28/gru_28/zeros/Const�
sequential_28/gru_28/zerosFill*sequential_28/gru_28/zeros/packed:output:0)sequential_28/gru_28/zeros/Const:output:0*
T0*'
_output_shapes
:���������02
sequential_28/gru_28/zeros�
#sequential_28/gru_28/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_28/gru_28/transpose/perm�
sequential_28/gru_28/transpose	Transposegru_28_input,sequential_28/gru_28/transpose/perm:output:0*
T0*+
_output_shapes
:���������2 
sequential_28/gru_28/transpose�
sequential_28/gru_28/Shape_1Shape"sequential_28/gru_28/transpose:y:0*
T0*
_output_shapes
:2
sequential_28/gru_28/Shape_1�
*sequential_28/gru_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_28/gru_28/strided_slice_1/stack�
,sequential_28/gru_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_28/gru_28/strided_slice_1/stack_1�
,sequential_28/gru_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_28/gru_28/strided_slice_1/stack_2�
$sequential_28/gru_28/strided_slice_1StridedSlice%sequential_28/gru_28/Shape_1:output:03sequential_28/gru_28/strided_slice_1/stack:output:05sequential_28/gru_28/strided_slice_1/stack_1:output:05sequential_28/gru_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_28/gru_28/strided_slice_1�
0sequential_28/gru_28/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������22
0sequential_28/gru_28/TensorArrayV2/element_shape�
"sequential_28/gru_28/TensorArrayV2TensorListReserve9sequential_28/gru_28/TensorArrayV2/element_shape:output:0-sequential_28/gru_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_28/gru_28/TensorArrayV2�
Jsequential_28/gru_28/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2L
Jsequential_28/gru_28/TensorArrayUnstack/TensorListFromTensor/element_shape�
<sequential_28/gru_28/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_28/gru_28/transpose:y:0Ssequential_28/gru_28/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_28/gru_28/TensorArrayUnstack/TensorListFromTensor�
*sequential_28/gru_28/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_28/gru_28/strided_slice_2/stack�
,sequential_28/gru_28/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_28/gru_28/strided_slice_2/stack_1�
,sequential_28/gru_28/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_28/gru_28/strided_slice_2/stack_2�
$sequential_28/gru_28/strided_slice_2StridedSlice"sequential_28/gru_28/transpose:y:03sequential_28/gru_28/strided_slice_2/stack:output:05sequential_28/gru_28/strided_slice_2/stack_1:output:05sequential_28/gru_28/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2&
$sequential_28/gru_28/strided_slice_2�
#sequential_28/gru_28/ReadVariableOpReadVariableOp,sequential_28_gru_28_readvariableop_resource*
_output_shapes
:	�*
dtype02%
#sequential_28/gru_28/ReadVariableOp�
sequential_28/gru_28/unstackUnpack+sequential_28/gru_28/ReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2
sequential_28/gru_28/unstack�
*sequential_28/gru_28/MatMul/ReadVariableOpReadVariableOp3sequential_28_gru_28_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02,
*sequential_28/gru_28/MatMul/ReadVariableOp�
sequential_28/gru_28/MatMulMatMul-sequential_28/gru_28/strided_slice_2:output:02sequential_28/gru_28/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_28/gru_28/MatMul�
sequential_28/gru_28/BiasAddBiasAdd%sequential_28/gru_28/MatMul:product:0%sequential_28/gru_28/unstack:output:0*
T0*(
_output_shapes
:����������2
sequential_28/gru_28/BiasAddz
sequential_28/gru_28/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
sequential_28/gru_28/Const�
$sequential_28/gru_28/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2&
$sequential_28/gru_28/split/split_dim�
sequential_28/gru_28/splitSplit-sequential_28/gru_28/split/split_dim:output:0%sequential_28/gru_28/BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
sequential_28/gru_28/split�
,sequential_28/gru_28/MatMul_1/ReadVariableOpReadVariableOp5sequential_28_gru_28_matmul_1_readvariableop_resource*
_output_shapes
:	0�*
dtype02.
,sequential_28/gru_28/MatMul_1/ReadVariableOp�
sequential_28/gru_28/MatMul_1MatMul#sequential_28/gru_28/zeros:output:04sequential_28/gru_28/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_28/gru_28/MatMul_1�
sequential_28/gru_28/BiasAdd_1BiasAdd'sequential_28/gru_28/MatMul_1:product:0%sequential_28/gru_28/unstack:output:1*
T0*(
_output_shapes
:����������2 
sequential_28/gru_28/BiasAdd_1�
sequential_28/gru_28/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2
sequential_28/gru_28/Const_1�
&sequential_28/gru_28/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2(
&sequential_28/gru_28/split_1/split_dim�
sequential_28/gru_28/split_1SplitV'sequential_28/gru_28/BiasAdd_1:output:0%sequential_28/gru_28/Const_1:output:0/sequential_28/gru_28/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
sequential_28/gru_28/split_1�
sequential_28/gru_28/addAddV2#sequential_28/gru_28/split:output:0%sequential_28/gru_28/split_1:output:0*
T0*'
_output_shapes
:���������02
sequential_28/gru_28/add�
sequential_28/gru_28/SigmoidSigmoidsequential_28/gru_28/add:z:0*
T0*'
_output_shapes
:���������02
sequential_28/gru_28/Sigmoid�
sequential_28/gru_28/add_1AddV2#sequential_28/gru_28/split:output:1%sequential_28/gru_28/split_1:output:1*
T0*'
_output_shapes
:���������02
sequential_28/gru_28/add_1�
sequential_28/gru_28/Sigmoid_1Sigmoidsequential_28/gru_28/add_1:z:0*
T0*'
_output_shapes
:���������02 
sequential_28/gru_28/Sigmoid_1�
sequential_28/gru_28/mulMul"sequential_28/gru_28/Sigmoid_1:y:0%sequential_28/gru_28/split_1:output:2*
T0*'
_output_shapes
:���������02
sequential_28/gru_28/mul�
sequential_28/gru_28/add_2AddV2#sequential_28/gru_28/split:output:2sequential_28/gru_28/mul:z:0*
T0*'
_output_shapes
:���������02
sequential_28/gru_28/add_2�
sequential_28/gru_28/TanhTanhsequential_28/gru_28/add_2:z:0*
T0*'
_output_shapes
:���������02
sequential_28/gru_28/Tanh�
sequential_28/gru_28/mul_1Mul sequential_28/gru_28/Sigmoid:y:0#sequential_28/gru_28/zeros:output:0*
T0*'
_output_shapes
:���������02
sequential_28/gru_28/mul_1}
sequential_28/gru_28/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sequential_28/gru_28/sub/x�
sequential_28/gru_28/subSub#sequential_28/gru_28/sub/x:output:0 sequential_28/gru_28/Sigmoid:y:0*
T0*'
_output_shapes
:���������02
sequential_28/gru_28/sub�
sequential_28/gru_28/mul_2Mulsequential_28/gru_28/sub:z:0sequential_28/gru_28/Tanh:y:0*
T0*'
_output_shapes
:���������02
sequential_28/gru_28/mul_2�
sequential_28/gru_28/add_3AddV2sequential_28/gru_28/mul_1:z:0sequential_28/gru_28/mul_2:z:0*
T0*'
_output_shapes
:���������02
sequential_28/gru_28/add_3�
2sequential_28/gru_28/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   24
2sequential_28/gru_28/TensorArrayV2_1/element_shape�
$sequential_28/gru_28/TensorArrayV2_1TensorListReserve;sequential_28/gru_28/TensorArrayV2_1/element_shape:output:0-sequential_28/gru_28/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_28/gru_28/TensorArrayV2_1x
sequential_28/gru_28/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_28/gru_28/time�
-sequential_28/gru_28/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2/
-sequential_28/gru_28/while/maximum_iterations�
'sequential_28/gru_28/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_28/gru_28/while/loop_counter�
sequential_28/gru_28/whileWhile0sequential_28/gru_28/while/loop_counter:output:06sequential_28/gru_28/while/maximum_iterations:output:0"sequential_28/gru_28/time:output:0-sequential_28/gru_28/TensorArrayV2_1:handle:0#sequential_28/gru_28/zeros:output:0-sequential_28/gru_28/strided_slice_1:output:0Lsequential_28/gru_28/TensorArrayUnstack/TensorListFromTensor:output_handle:0,sequential_28_gru_28_readvariableop_resource3sequential_28_gru_28_matmul_readvariableop_resource5sequential_28_gru_28_matmul_1_readvariableop_resource+^sequential_28/gru_28/MatMul/ReadVariableOp-^sequential_28/gru_28/MatMul_1/ReadVariableOp$^sequential_28/gru_28/ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������0: : : : : *0
body(R&
$sequential_28_gru_28_while_body_1456*0
cond(R&
$sequential_28_gru_28_while_cond_1455*8
output_shapes'
%: : : : :���������0: : : : : *
parallel_iterations 2
sequential_28/gru_28/while�
Esequential_28/gru_28/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   2G
Esequential_28/gru_28/TensorArrayV2Stack/TensorListStack/element_shape�
7sequential_28/gru_28/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_28/gru_28/while:output:3Nsequential_28/gru_28/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������0*
element_dtype029
7sequential_28/gru_28/TensorArrayV2Stack/TensorListStack�
*sequential_28/gru_28/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2,
*sequential_28/gru_28/strided_slice_3/stack�
,sequential_28/gru_28/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_28/gru_28/strided_slice_3/stack_1�
,sequential_28/gru_28/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_28/gru_28/strided_slice_3/stack_2�
$sequential_28/gru_28/strided_slice_3StridedSlice@sequential_28/gru_28/TensorArrayV2Stack/TensorListStack:tensor:03sequential_28/gru_28/strided_slice_3/stack:output:05sequential_28/gru_28/strided_slice_3/stack_1:output:05sequential_28/gru_28/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������0*
shrink_axis_mask2&
$sequential_28/gru_28/strided_slice_3�
%sequential_28/gru_28/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_28/gru_28/transpose_1/perm�
 sequential_28/gru_28/transpose_1	Transpose@sequential_28/gru_28/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_28/gru_28/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������02"
 sequential_28/gru_28/transpose_1�
,sequential_28/dense_84/MatMul/ReadVariableOpReadVariableOp5sequential_28_dense_84_matmul_readvariableop_resource*
_output_shapes
:	0�*
dtype02.
,sequential_28/dense_84/MatMul/ReadVariableOp�
sequential_28/dense_84/MatMulMatMul-sequential_28/gru_28/strided_slice_3:output:04sequential_28/dense_84/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_28/dense_84/MatMul�
-sequential_28/dense_84/BiasAdd/ReadVariableOpReadVariableOp6sequential_28_dense_84_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_28/dense_84/BiasAdd/ReadVariableOp�
sequential_28/dense_84/BiasAddBiasAdd'sequential_28/dense_84/MatMul:product:05sequential_28/dense_84/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
sequential_28/dense_84/BiasAdd�
sequential_28/dense_84/ReluRelu'sequential_28/dense_84/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_28/dense_84/Relu�
,sequential_28/dense_85/MatMul/ReadVariableOpReadVariableOp5sequential_28_dense_85_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02.
,sequential_28/dense_85/MatMul/ReadVariableOp�
sequential_28/dense_85/MatMulMatMul)sequential_28/dense_84/Relu:activations:04sequential_28/dense_85/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
sequential_28/dense_85/MatMul�
-sequential_28/dense_85/BiasAdd/ReadVariableOpReadVariableOp6sequential_28_dense_85_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02/
-sequential_28/dense_85/BiasAdd/ReadVariableOp�
sequential_28/dense_85/BiasAddBiasAdd'sequential_28/dense_85/MatMul:product:05sequential_28/dense_85/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2 
sequential_28/dense_85/BiasAdd�
sequential_28/dense_85/ReluRelu'sequential_28/dense_85/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
sequential_28/dense_85/Relu�
,sequential_28/dense_86/MatMul/ReadVariableOpReadVariableOp5sequential_28_dense_86_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02.
,sequential_28/dense_86/MatMul/ReadVariableOp�
sequential_28/dense_86/MatMulMatMul)sequential_28/dense_85/Relu:activations:04sequential_28/dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_28/dense_86/MatMul�
-sequential_28/dense_86/BiasAdd/ReadVariableOpReadVariableOp6sequential_28_dense_86_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_28/dense_86/BiasAdd/ReadVariableOp�
sequential_28/dense_86/BiasAddBiasAdd'sequential_28/dense_86/MatMul:product:05sequential_28/dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
sequential_28/dense_86/BiasAdd�
IdentityIdentity'sequential_28/dense_86/BiasAdd:output:0.^sequential_28/dense_84/BiasAdd/ReadVariableOp-^sequential_28/dense_84/MatMul/ReadVariableOp.^sequential_28/dense_85/BiasAdd/ReadVariableOp-^sequential_28/dense_85/MatMul/ReadVariableOp.^sequential_28/dense_86/BiasAdd/ReadVariableOp-^sequential_28/dense_86/MatMul/ReadVariableOp+^sequential_28/gru_28/MatMul/ReadVariableOp-^sequential_28/gru_28/MatMul_1/ReadVariableOp$^sequential_28/gru_28/ReadVariableOp^sequential_28/gru_28/while*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������:::::::::2^
-sequential_28/dense_84/BiasAdd/ReadVariableOp-sequential_28/dense_84/BiasAdd/ReadVariableOp2\
,sequential_28/dense_84/MatMul/ReadVariableOp,sequential_28/dense_84/MatMul/ReadVariableOp2^
-sequential_28/dense_85/BiasAdd/ReadVariableOp-sequential_28/dense_85/BiasAdd/ReadVariableOp2\
,sequential_28/dense_85/MatMul/ReadVariableOp,sequential_28/dense_85/MatMul/ReadVariableOp2^
-sequential_28/dense_86/BiasAdd/ReadVariableOp-sequential_28/dense_86/BiasAdd/ReadVariableOp2\
,sequential_28/dense_86/MatMul/ReadVariableOp,sequential_28/dense_86/MatMul/ReadVariableOp2X
*sequential_28/gru_28/MatMul/ReadVariableOp*sequential_28/gru_28/MatMul/ReadVariableOp2\
,sequential_28/gru_28/MatMul_1/ReadVariableOp,sequential_28/gru_28/MatMul_1/ReadVariableOp2J
#sequential_28/gru_28/ReadVariableOp#sequential_28/gru_28/ReadVariableOp28
sequential_28/gru_28/whilesequential_28/gru_28/while:, (
&
_user_specified_namegru_28_input
�
�
while_cond_3541
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1,
(while_cond_3541___redundant_placeholder0,
(while_cond_3541___redundant_placeholder1,
(while_cond_3541___redundant_placeholder2,
(while_cond_3541___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :���������0: ::::
�P
�
@__inference_gru_28_layer_call_and_return_conditional_losses_3141
inputs_0
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource
identity��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :02
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :02
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������02
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanhd
mul_1MulSigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcematmul_readvariableop_resource matmul_1_readvariableop_resource^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������0: : : : : *
bodyR
while_body_3052*
condR
while_cond_3051*8
output_shapes'
%: : : : :���������0: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����0   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������0*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������0*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������02
transpose_1�
IdentityIdentitystrided_slice_3:output:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp^while*
T0*'
_output_shapes
:���������02

Identity"
identityIdentity:output:0*?
_input_shapes.
,:������������������:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2
whilewhile:( $
"
_user_specified_name
inputs/0
�
�
G__inference_sequential_28_layer_call_and_return_conditional_losses_2564

inputs)
%gru_28_statefulpartitionedcall_args_1)
%gru_28_statefulpartitionedcall_args_2)
%gru_28_statefulpartitionedcall_args_3+
'dense_84_statefulpartitionedcall_args_1+
'dense_84_statefulpartitionedcall_args_2+
'dense_85_statefulpartitionedcall_args_1+
'dense_85_statefulpartitionedcall_args_2+
'dense_86_statefulpartitionedcall_args_1+
'dense_86_statefulpartitionedcall_args_2
identity�� dense_84/StatefulPartitionedCall� dense_85/StatefulPartitionedCall� dense_86/StatefulPartitionedCall�gru_28/StatefulPartitionedCall�
gru_28/StatefulPartitionedCallStatefulPartitionedCallinputs%gru_28_statefulpartitionedcall_args_1%gru_28_statefulpartitionedcall_args_2%gru_28_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������0**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_gru_28_layer_call_and_return_conditional_losses_24062 
gru_28/StatefulPartitionedCall�
 dense_84/StatefulPartitionedCallStatefulPartitionedCall'gru_28/StatefulPartitionedCall:output:0'dense_84_statefulpartitionedcall_args_1'dense_84_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_84_layer_call_and_return_conditional_losses_24382"
 dense_84/StatefulPartitionedCall�
 dense_85/StatefulPartitionedCallStatefulPartitionedCall)dense_84/StatefulPartitionedCall:output:0'dense_85_statefulpartitionedcall_args_1'dense_85_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_85_layer_call_and_return_conditional_losses_24612"
 dense_85/StatefulPartitionedCall�
 dense_86/StatefulPartitionedCallStatefulPartitionedCall)dense_85/StatefulPartitionedCall:output:0'dense_86_statefulpartitionedcall_args_1'dense_86_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_86_layer_call_and_return_conditional_losses_24832"
 dense_86/StatefulPartitionedCall�
IdentityIdentity)dense_86/StatefulPartitionedCall:output:0!^dense_84/StatefulPartitionedCall!^dense_85/StatefulPartitionedCall!^dense_86/StatefulPartitionedCall^gru_28/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������:::::::::2D
 dense_84/StatefulPartitionedCall dense_84/StatefulPartitionedCall2D
 dense_85/StatefulPartitionedCall dense_85/StatefulPartitionedCall2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2@
gru_28/StatefulPartitionedCallgru_28/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
B__inference_dense_85_layer_call_and_return_conditional_losses_3676

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
'__inference_dense_86_layer_call_fn_3700

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_86_layer_call_and_return_conditional_losses_24832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
gru_28_while_cond_2845
gru_28_while_loop_counter#
gru_28_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_gru_28_strided_slice_13
/gru_28_while_cond_2845___redundant_placeholder03
/gru_28_while_cond_2845___redundant_placeholder13
/gru_28_while_cond_2845___redundant_placeholder23
/gru_28_while_cond_2845___redundant_placeholder3
identity
_
LessLessplaceholderless_gru_28_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :���������0: ::::
�1
�
while_body_3542
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�ReadVariableOp�
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   23
1TensorArrayV2Read/TensorListGetItem/element_shape�
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem{
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes
:	�*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:�:�*	
num2	
unstack�
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOp�
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:����������2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2
split�
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	0�*
dtype02
MatMul_1/ReadVariableOp�
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:����������2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"0   0   ����2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
split_1/split_dim�
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:���������0:���������0:���������0*
	num_split2	
split_1g
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:���������02
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������02	
Sigmoidk
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:���������02
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:���������02
	Sigmoid_1d
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:���������02
mulb
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:���������02
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:���������02
Tanhc
mul_1MulSigmoid:y:0placeholder_2*
T0*'
_output_shapes
:���������02
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:���������02
subZ
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:���������02
mul_2_
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:���������02
add_3�
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_3:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_4/yW
add_4AddV2placeholderadd_4/y:output:0*
T0*
_output_shapes
: 2
add_4T
add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_5/y^
add_5AddV2while_loop_counteradd_5/y:output:0*
T0*
_output_shapes
: 2
add_5�
IdentityIdentity	add_5:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity�

Identity_1Identitywhile_maximum_iterations^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1�

Identity_2Identity	add_4:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2�

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3�

Identity_4Identity	add_3:z:0^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
T0*'
_output_shapes
:���������02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"�
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :���������0: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp
�
�
while_cond_2020
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1,
(while_cond_2020___redundant_placeholder0,
(while_cond_2020___redundant_placeholder1,
(while_cond_2020___redundant_placeholder2,
(while_cond_2020___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :���������0: ::::
�
�
,__inference_sequential_28_layer_call_fn_2545
gru_28_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgru_28_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9*
Tin
2
*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_28_layer_call_and_return_conditional_losses_25332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:, (
&
_user_specified_namegru_28_input
�
�
B__inference_dense_86_layer_call_and_return_conditional_losses_2483

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�

�
,__inference_sequential_28_layer_call_fn_2983

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9*
Tin
2
*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_28_layer_call_and_return_conditional_losses_25642
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�

�
,__inference_sequential_28_layer_call_fn_2969

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9*
Tin
2
*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_28_layer_call_and_return_conditional_losses_25332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
gru_28_while_cond_2667
gru_28_while_loop_counter#
gru_28_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_gru_28_strided_slice_13
/gru_28_while_cond_2667___redundant_placeholder03
/gru_28_while_cond_2667___redundant_placeholder13
/gru_28_while_cond_2667___redundant_placeholder23
/gru_28_while_cond_2667___redundant_placeholder3
identity
_
LessLessplaceholderless_gru_28_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :���������0: ::::
�
�
%__inference_gru_28_layer_call_fn_3639

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������0**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_gru_28_layer_call_and_return_conditional_losses_22482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������02

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
I
gru_28_input9
serving_default_gru_28_input:0���������<
dense_860
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�+
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
Z__call__
[_default_save_signature
*\&call_and_return_all_conditional_losses"�(
_tf_keras_sequential�({"class_name": "Sequential", "name": "sequential_28", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_28", "layers": [{"class_name": "GRU", "config": {"name": "gru_28", "trainable": true, "batch_input_shape": [null, 3, 1], "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 48, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense_84", "trainable": true, "dtype": "float32", "units": 168, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_85", "trainable": true, "dtype": "float32", "units": 336, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_86", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 1], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_28", "layers": [{"class_name": "GRU", "config": {"name": "gru_28", "trainable": true, "batch_input_shape": [null, 3, 1], "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 48, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense_84", "trainable": true, "dtype": "float32", "units": 168, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_85", "trainable": true, "dtype": "float32", "units": 336, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_86", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mae", "metrics": [], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "gru_28_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 3, 1], "config": {"batch_input_shape": [null, 3, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "gru_28_input"}}
�

cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"�	
_tf_keras_layer�{"class_name": "GRU", "name": "gru_28", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 3, 1], "config": {"name": "gru_28", "trainable": true, "batch_input_shape": [null, 3, 1], "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 48, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 1], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}]}
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
___call__
*`&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_84", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_84", "trainable": true, "dtype": "float32", "units": 168, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 48}}}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
a__call__
*b&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_85", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_85", "trainable": true, "dtype": "float32", "units": 336, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 168}}}}
�

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
c__call__
*d&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_86", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_86", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 336}}}}
�
$iter

%beta_1

&beta_2
	'decay
(learning_ratemHmImJmKmLmM)mN*mO+mPvQvRvSvTvUvV)vW*vX+vY"
	optimizer
_
)0
*1
+2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
_
)0
*1
+2
3
4
5
6
7
8"
trackable_list_wrapper
�
,layer_regularization_losses
	variables
-metrics
.non_trainable_variables
regularization_losses

/layers
	trainable_variables
Z__call__
[_default_save_signature
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
,
eserving_default"
signature_map
�

)kernel
*recurrent_kernel
+bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
f__call__
*g&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "GRUCell", "name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 48, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
�
4layer_regularization_losses
	variables
5metrics
6non_trainable_variables
regularization_losses

7layers
trainable_variables
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
": 	0�2dense_84/kernel
:�2dense_84/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
8layer_regularization_losses
	variables
9metrics
:non_trainable_variables
regularization_losses

;layers
trainable_variables
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
#:!
��2dense_85/kernel
:�2dense_85/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
<layer_regularization_losses
	variables
=metrics
>non_trainable_variables
regularization_losses

?layers
trainable_variables
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
": 	�2dense_86/kernel
:2dense_86/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
@layer_regularization_losses
 	variables
Ametrics
Bnon_trainable_variables
!regularization_losses

Clayers
"trainable_variables
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
:	 (2training/Adam/iter
: (2training/Adam/beta_1
: (2training/Adam/beta_2
: (2training/Adam/decay
%:# (2training/Adam/learning_rate
 :	�2gru_28/kernel
*:(	0�2gru_28/recurrent_kernel
:	�2gru_28/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
�
Dlayer_regularization_losses
0	variables
Emetrics
Fnon_trainable_variables
1regularization_losses

Glayers
2trainable_variables
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0:.	0�2training/Adam/dense_84/kernel/m
*:(�2training/Adam/dense_84/bias/m
1:/
��2training/Adam/dense_85/kernel/m
*:(�2training/Adam/dense_85/bias/m
0:.	�2training/Adam/dense_86/kernel/m
):'2training/Adam/dense_86/bias/m
.:,	�2training/Adam/gru_28/kernel/m
8:6	0�2'training/Adam/gru_28/recurrent_kernel/m
,:*	�2training/Adam/gru_28/bias/m
0:.	0�2training/Adam/dense_84/kernel/v
*:(�2training/Adam/dense_84/bias/v
1:/
��2training/Adam/dense_85/kernel/v
*:(�2training/Adam/dense_85/bias/v
0:.	�2training/Adam/dense_86/kernel/v
):'2training/Adam/dense_86/bias/v
.:,	�2training/Adam/gru_28/kernel/v
8:6	0�2'training/Adam/gru_28/recurrent_kernel/v
,:*	�2training/Adam/gru_28/bias/v
�2�
,__inference_sequential_28_layer_call_fn_2545
,__inference_sequential_28_layer_call_fn_2969
,__inference_sequential_28_layer_call_fn_2576
,__inference_sequential_28_layer_call_fn_2983�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference__wrapped_model_1565�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� */�,
*�'
gru_28_input���������
�2�
G__inference_sequential_28_layer_call_and_return_conditional_losses_2955
G__inference_sequential_28_layer_call_and_return_conditional_losses_2513
G__inference_sequential_28_layer_call_and_return_conditional_losses_2777
G__inference_sequential_28_layer_call_and_return_conditional_losses_2496�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
%__inference_gru_28_layer_call_fn_3315
%__inference_gru_28_layer_call_fn_3307
%__inference_gru_28_layer_call_fn_3647
%__inference_gru_28_layer_call_fn_3639�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
@__inference_gru_28_layer_call_and_return_conditional_losses_3299
@__inference_gru_28_layer_call_and_return_conditional_losses_3141
@__inference_gru_28_layer_call_and_return_conditional_losses_3473
@__inference_gru_28_layer_call_and_return_conditional_losses_3631�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_dense_84_layer_call_fn_3665�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_84_layer_call_and_return_conditional_losses_3658�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_85_layer_call_fn_3683�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_85_layer_call_and_return_conditional_losses_3676�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_86_layer_call_fn_3700�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_86_layer_call_and_return_conditional_losses_3693�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
6B4
"__inference_signature_wrapper_2599gru_28_input
�2�
'__inference_gru_cell_layer_call_fn_3802
'__inference_gru_cell_layer_call_fn_3791�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_gru_cell_layer_call_and_return_conditional_losses_3740
B__inference_gru_cell_layer_call_and_return_conditional_losses_3780�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 �
__inference__wrapped_model_1565{	+)*9�6
/�,
*�'
gru_28_input���������
� "3�0
.
dense_86"�
dense_86����������
B__inference_dense_84_layer_call_and_return_conditional_losses_3658]/�,
%�"
 �
inputs���������0
� "&�#
�
0����������
� {
'__inference_dense_84_layer_call_fn_3665P/�,
%�"
 �
inputs���������0
� "������������
B__inference_dense_85_layer_call_and_return_conditional_losses_3676^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
'__inference_dense_85_layer_call_fn_3683Q0�-
&�#
!�
inputs����������
� "������������
B__inference_dense_86_layer_call_and_return_conditional_losses_3693]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� {
'__inference_dense_86_layer_call_fn_3700P0�-
&�#
!�
inputs����������
� "�����������
@__inference_gru_28_layer_call_and_return_conditional_losses_3141}+)*O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "%�"
�
0���������0
� �
@__inference_gru_28_layer_call_and_return_conditional_losses_3299}+)*O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "%�"
�
0���������0
� �
@__inference_gru_28_layer_call_and_return_conditional_losses_3473m+)*?�<
5�2
$�!
inputs���������

 
p

 
� "%�"
�
0���������0
� �
@__inference_gru_28_layer_call_and_return_conditional_losses_3631m+)*?�<
5�2
$�!
inputs���������

 
p 

 
� "%�"
�
0���������0
� �
%__inference_gru_28_layer_call_fn_3307p+)*O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "����������0�
%__inference_gru_28_layer_call_fn_3315p+)*O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "����������0�
%__inference_gru_28_layer_call_fn_3639`+)*?�<
5�2
$�!
inputs���������

 
p

 
� "����������0�
%__inference_gru_28_layer_call_fn_3647`+)*?�<
5�2
$�!
inputs���������

 
p 

 
� "����������0�
B__inference_gru_cell_layer_call_and_return_conditional_losses_3740�+)*\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������0
p
� "R�O
H�E
�
0/0���������0
$�!
�
0/1/0���������0
� �
B__inference_gru_cell_layer_call_and_return_conditional_losses_3780�+)*\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������0
p 
� "R�O
H�E
�
0/0���������0
$�!
�
0/1/0���������0
� �
'__inference_gru_cell_layer_call_fn_3791�+)*\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������0
p
� "D�A
�
0���������0
"�
�
1/0���������0�
'__inference_gru_cell_layer_call_fn_3802�+)*\�Y
R�O
 �
inputs���������
'�$
"�
states/0���������0
p 
� "D�A
�
0���������0
"�
�
1/0���������0�
G__inference_sequential_28_layer_call_and_return_conditional_losses_2496u	+)*A�>
7�4
*�'
gru_28_input���������
p

 
� "%�"
�
0���������
� �
G__inference_sequential_28_layer_call_and_return_conditional_losses_2513u	+)*A�>
7�4
*�'
gru_28_input���������
p 

 
� "%�"
�
0���������
� �
G__inference_sequential_28_layer_call_and_return_conditional_losses_2777o	+)*;�8
1�.
$�!
inputs���������
p

 
� "%�"
�
0���������
� �
G__inference_sequential_28_layer_call_and_return_conditional_losses_2955o	+)*;�8
1�.
$�!
inputs���������
p 

 
� "%�"
�
0���������
� �
,__inference_sequential_28_layer_call_fn_2545h	+)*A�>
7�4
*�'
gru_28_input���������
p

 
� "�����������
,__inference_sequential_28_layer_call_fn_2576h	+)*A�>
7�4
*�'
gru_28_input���������
p 

 
� "�����������
,__inference_sequential_28_layer_call_fn_2969b	+)*;�8
1�.
$�!
inputs���������
p

 
� "�����������
,__inference_sequential_28_layer_call_fn_2983b	+)*;�8
1�.
$�!
inputs���������
p 

 
� "�����������
"__inference_signature_wrapper_2599�	+)*I�F
� 
?�<
:
gru_28_input*�'
gru_28_input���������"3�0
.
dense_86"�
dense_86���������