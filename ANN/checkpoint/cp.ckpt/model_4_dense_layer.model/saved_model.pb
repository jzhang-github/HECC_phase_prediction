Мы
жў
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
╛
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8щ╙
Ж
Hidden_layer_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameHidden_layer_0/kernel

)Hidden_layer_0/kernel/Read/ReadVariableOpReadVariableOpHidden_layer_0/kernel*
_output_shapes

:*
dtype0
~
Hidden_layer_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameHidden_layer_0/bias
w
'Hidden_layer_0/bias/Read/ReadVariableOpReadVariableOpHidden_layer_0/bias*
_output_shapes
:*
dtype0
Ж
Hidden_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameHidden_layer_1/kernel

)Hidden_layer_1/kernel/Read/ReadVariableOpReadVariableOpHidden_layer_1/kernel*
_output_shapes

:*
dtype0
~
Hidden_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameHidden_layer_1/bias
w
'Hidden_layer_1/bias/Read/ReadVariableOpReadVariableOpHidden_layer_1/bias*
_output_shapes
:*
dtype0
В
Output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameOutput_layer/kernel
{
'Output_layer/kernel/Read/ReadVariableOpReadVariableOpOutput_layer/kernel*
_output_shapes

:*
dtype0
z
Output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameOutput_layer/bias
s
%Output_layer/bias/Read/ReadVariableOpReadVariableOpOutput_layer/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
Ф
Adam/Hidden_layer_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/Hidden_layer_0/kernel/m
Н
0Adam/Hidden_layer_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_0/kernel/m*
_output_shapes

:*
dtype0
М
Adam/Hidden_layer_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/Hidden_layer_0/bias/m
Е
.Adam/Hidden_layer_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_0/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/Hidden_layer_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/Hidden_layer_1/kernel/m
Н
0Adam/Hidden_layer_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_1/kernel/m*
_output_shapes

:*
dtype0
М
Adam/Hidden_layer_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/Hidden_layer_1/bias/m
Е
.Adam/Hidden_layer_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_1/bias/m*
_output_shapes
:*
dtype0
Р
Adam/Output_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/Output_layer/kernel/m
Й
.Adam/Output_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Output_layer/kernel/m*
_output_shapes

:*
dtype0
И
Adam/Output_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Output_layer/bias/m
Б
,Adam/Output_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/Output_layer/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/Hidden_layer_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/Hidden_layer_0/kernel/v
Н
0Adam/Hidden_layer_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_0/kernel/v*
_output_shapes

:*
dtype0
М
Adam/Hidden_layer_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/Hidden_layer_0/bias/v
Е
.Adam/Hidden_layer_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_0/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/Hidden_layer_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/Hidden_layer_1/kernel/v
Н
0Adam/Hidden_layer_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_1/kernel/v*
_output_shapes

:*
dtype0
М
Adam/Hidden_layer_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/Hidden_layer_1/bias/v
Е
.Adam/Hidden_layer_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_layer_1/bias/v*
_output_shapes
:*
dtype0
Р
Adam/Output_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/Output_layer/kernel/v
Й
.Adam/Output_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Output_layer/kernel/v*
_output_shapes

:*
dtype0
И
Adam/Output_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Output_layer/bias/v
Б
,Adam/Output_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/Output_layer/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
М&
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╟%
value╜%B║% B│%
ц
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
h


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
м
iter

beta_1

beta_2
	decay
 learning_rate
m@mAmBmCmDmE
vFvGvHvIvJvK
 
*

0
1
2
3
4
5
*

0
1
2
3
4
5
н
regularization_losses
!layer_metrics
	variables

"layers
#non_trainable_variables
$metrics
%layer_regularization_losses
trainable_variables
 
a_
VARIABLE_VALUEHidden_layer_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEHidden_layer_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 


0
1


0
1
н
regularization_losses
&layer_metrics
'non_trainable_variables

(layers
	variables
)metrics
*layer_regularization_losses
trainable_variables
a_
VARIABLE_VALUEHidden_layer_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEHidden_layer_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
н
regularization_losses
+layer_metrics
,non_trainable_variables

-layers
	variables
.metrics
/layer_regularization_losses
trainable_variables
_]
VARIABLE_VALUEOutput_layer/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEOutput_layer/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
н
regularization_losses
0layer_metrics
1non_trainable_variables

2layers
	variables
3metrics
4layer_regularization_losses
trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
 

50
61
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
4
	7total
	8count
9	variables
:	keras_api
D
	;total
	<count
=
_fn_kwargs
>	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

70
81

9	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1

>	variables
ЕВ
VARIABLE_VALUEAdam/Hidden_layer_0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/Hidden_layer_0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdam/Hidden_layer_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/Hidden_layer_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEAdam/Output_layer/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Output_layer/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdam/Hidden_layer_0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/Hidden_layer_0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdam/Hidden_layer_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/Hidden_layer_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEAdam/Output_layer/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Output_layer/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_Input_layerPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
┴
StatefulPartitionedCallStatefulPartitionedCallserving_default_Input_layerHidden_layer_0/kernelHidden_layer_0/biasHidden_layer_1/kernelHidden_layer_1/biasOutput_layer/kernelOutput_layer/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_21041
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
К
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)Hidden_layer_0/kernel/Read/ReadVariableOp'Hidden_layer_0/bias/Read/ReadVariableOp)Hidden_layer_1/kernel/Read/ReadVariableOp'Hidden_layer_1/bias/Read/ReadVariableOp'Output_layer/kernel/Read/ReadVariableOp%Output_layer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp0Adam/Hidden_layer_0/kernel/m/Read/ReadVariableOp.Adam/Hidden_layer_0/bias/m/Read/ReadVariableOp0Adam/Hidden_layer_1/kernel/m/Read/ReadVariableOp.Adam/Hidden_layer_1/bias/m/Read/ReadVariableOp.Adam/Output_layer/kernel/m/Read/ReadVariableOp,Adam/Output_layer/bias/m/Read/ReadVariableOp0Adam/Hidden_layer_0/kernel/v/Read/ReadVariableOp.Adam/Hidden_layer_0/bias/v/Read/ReadVariableOp0Adam/Hidden_layer_1/kernel/v/Read/ReadVariableOp.Adam/Hidden_layer_1/bias/v/Read/ReadVariableOp.Adam/Output_layer/kernel/v/Read/ReadVariableOp,Adam/Output_layer/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__traced_save_21289
щ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameHidden_layer_0/kernelHidden_layer_0/biasHidden_layer_1/kernelHidden_layer_1/biasOutput_layer/kernelOutput_layer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/Hidden_layer_0/kernel/mAdam/Hidden_layer_0/bias/mAdam/Hidden_layer_1/kernel/mAdam/Hidden_layer_1/bias/mAdam/Output_layer/kernel/mAdam/Output_layer/bias/mAdam/Hidden_layer_0/kernel/vAdam/Hidden_layer_0/bias/vAdam/Hidden_layer_1/kernel/vAdam/Hidden_layer_1/bias/vAdam/Output_layer/kernel/vAdam/Output_layer/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_restore_21380═с
√	
т
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_21136

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ї
ї
E__inference_sequential_layer_call_and_return_conditional_losses_20963

inputs
hidden_layer_0_20947
hidden_layer_0_20949
hidden_layer_1_20952
hidden_layer_1_20954
output_layer_20957
output_layer_20959
identityИв&Hidden_layer_0/StatefulPartitionedCallв&Hidden_layer_1/StatefulPartitionedCallв$Output_layer/StatefulPartitionedCallп
&Hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_layer_0_20947hidden_layer_0_20949*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_208512(
&Hidden_layer_0/StatefulPartitionedCall╪
&Hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_20952hidden_layer_1_20954*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_208782(
&Hidden_layer_1/StatefulPartitionedCall╬
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_1/StatefulPartitionedCall:output:0output_layer_20957output_layer_20959*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_Output_layer_layer_call_and_return_conditional_losses_209052&
$Output_layer/StatefulPartitionedCall·
IdentityIdentity-Output_layer/StatefulPartitionedCall:output:0'^Hidden_layer_0/StatefulPartitionedCall'^Hidden_layer_1/StatefulPartitionedCall%^Output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2P
&Hidden_layer_0/StatefulPartitionedCall&Hidden_layer_0/StatefulPartitionedCall2P
&Hidden_layer_1/StatefulPartitionedCall&Hidden_layer_1/StatefulPartitionedCall2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
∙	
р
G__inference_Output_layer_layer_call_and_return_conditional_losses_20905

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╝!
■
E__inference_sequential_layer_call_and_return_conditional_losses_21091

inputs1
-hidden_layer_0_matmul_readvariableop_resource2
.hidden_layer_0_biasadd_readvariableop_resource1
-hidden_layer_1_matmul_readvariableop_resource2
.hidden_layer_1_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identityИв%Hidden_layer_0/BiasAdd/ReadVariableOpв$Hidden_layer_0/MatMul/ReadVariableOpв%Hidden_layer_1/BiasAdd/ReadVariableOpв$Hidden_layer_1/MatMul/ReadVariableOpв#Output_layer/BiasAdd/ReadVariableOpв"Output_layer/MatMul/ReadVariableOp║
$Hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$Hidden_layer_0/MatMul/ReadVariableOpа
Hidden_layer_0/MatMulMatMulinputs,Hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Hidden_layer_0/MatMul╣
%Hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Hidden_layer_0/BiasAdd/ReadVariableOp╜
Hidden_layer_0/BiasAddBiasAddHidden_layer_0/MatMul:product:0-Hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Hidden_layer_0/BiasAddО
Hidden_layer_0/SoftmaxSoftmaxHidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:         2
Hidden_layer_0/Softmax║
$Hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$Hidden_layer_1/MatMul/ReadVariableOp║
Hidden_layer_1/MatMulMatMul Hidden_layer_0/Softmax:softmax:0,Hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Hidden_layer_1/MatMul╣
%Hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Hidden_layer_1/BiasAdd/ReadVariableOp╜
Hidden_layer_1/BiasAddBiasAddHidden_layer_1/MatMul:product:0-Hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Hidden_layer_1/BiasAddО
Hidden_layer_1/SoftmaxSoftmaxHidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
Hidden_layer_1/Softmax┤
"Output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"Output_layer/MatMul/ReadVariableOp┤
Output_layer/MatMulMatMul Hidden_layer_1/Softmax:softmax:0*Output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Output_layer/MatMul│
#Output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#Output_layer/BiasAdd/ReadVariableOp╡
Output_layer/BiasAddBiasAddOutput_layer/MatMul:product:0+Output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Output_layer/BiasAddИ
Output_layer/SoftmaxSoftmaxOutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:         2
Output_layer/Softmax█
IdentityIdentityOutput_layer/Softmax:softmax:0&^Hidden_layer_0/BiasAdd/ReadVariableOp%^Hidden_layer_0/MatMul/ReadVariableOp&^Hidden_layer_1/BiasAdd/ReadVariableOp%^Hidden_layer_1/MatMul/ReadVariableOp$^Output_layer/BiasAdd/ReadVariableOp#^Output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2N
%Hidden_layer_0/BiasAdd/ReadVariableOp%Hidden_layer_0/BiasAdd/ReadVariableOp2L
$Hidden_layer_0/MatMul/ReadVariableOp$Hidden_layer_0/MatMul/ReadVariableOp2N
%Hidden_layer_1/BiasAdd/ReadVariableOp%Hidden_layer_1/BiasAdd/ReadVariableOp2L
$Hidden_layer_1/MatMul/ReadVariableOp$Hidden_layer_1/MatMul/ReadVariableOp2J
#Output_layer/BiasAdd/ReadVariableOp#Output_layer/BiasAdd/ReadVariableOp2H
"Output_layer/MatMul/ReadVariableOp"Output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
▀
╗
*__inference_sequential_layer_call_fn_21125

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_209992
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Д
·
E__inference_sequential_layer_call_and_return_conditional_losses_20941
input_layer
hidden_layer_0_20925
hidden_layer_0_20927
hidden_layer_1_20930
hidden_layer_1_20932
output_layer_20935
output_layer_20937
identityИв&Hidden_layer_0/StatefulPartitionedCallв&Hidden_layer_1/StatefulPartitionedCallв$Output_layer/StatefulPartitionedCall┤
&Hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_layer_0_20925hidden_layer_0_20927*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_208512(
&Hidden_layer_0/StatefulPartitionedCall╪
&Hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_20930hidden_layer_1_20932*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_208782(
&Hidden_layer_1/StatefulPartitionedCall╬
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_1/StatefulPartitionedCall:output:0output_layer_20935output_layer_20937*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_Output_layer_layer_call_and_return_conditional_losses_209052&
$Output_layer/StatefulPartitionedCall·
IdentityIdentity-Output_layer/StatefulPartitionedCall:output:0'^Hidden_layer_0/StatefulPartitionedCall'^Hidden_layer_1/StatefulPartitionedCall%^Output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2P
&Hidden_layer_0/StatefulPartitionedCall&Hidden_layer_0/StatefulPartitionedCall2P
&Hidden_layer_1/StatefulPartitionedCall&Hidden_layer_1/StatefulPartitionedCall2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall:T P
'
_output_shapes
:         
%
_user_specified_nameInput_layer
ч
Г
.__inference_Hidden_layer_1_layer_call_fn_21165

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_208782
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
√	
т
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_20878

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
б(
т
 __inference__wrapped_model_20836
input_layer<
8sequential_hidden_layer_0_matmul_readvariableop_resource=
9sequential_hidden_layer_0_biasadd_readvariableop_resource<
8sequential_hidden_layer_1_matmul_readvariableop_resource=
9sequential_hidden_layer_1_biasadd_readvariableop_resource:
6sequential_output_layer_matmul_readvariableop_resource;
7sequential_output_layer_biasadd_readvariableop_resource
identityИв0sequential/Hidden_layer_0/BiasAdd/ReadVariableOpв/sequential/Hidden_layer_0/MatMul/ReadVariableOpв0sequential/Hidden_layer_1/BiasAdd/ReadVariableOpв/sequential/Hidden_layer_1/MatMul/ReadVariableOpв.sequential/Output_layer/BiasAdd/ReadVariableOpв-sequential/Output_layer/MatMul/ReadVariableOp█
/sequential/Hidden_layer_0/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype021
/sequential/Hidden_layer_0/MatMul/ReadVariableOp╞
 sequential/Hidden_layer_0/MatMulMatMulinput_layer7sequential/Hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2"
 sequential/Hidden_layer_0/MatMul┌
0sequential/Hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0sequential/Hidden_layer_0/BiasAdd/ReadVariableOpщ
!sequential/Hidden_layer_0/BiasAddBiasAdd*sequential/Hidden_layer_0/MatMul:product:08sequential/Hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2#
!sequential/Hidden_layer_0/BiasAddп
!sequential/Hidden_layer_0/SoftmaxSoftmax*sequential/Hidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:         2#
!sequential/Hidden_layer_0/Softmax█
/sequential/Hidden_layer_1/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype021
/sequential/Hidden_layer_1/MatMul/ReadVariableOpц
 sequential/Hidden_layer_1/MatMulMatMul+sequential/Hidden_layer_0/Softmax:softmax:07sequential/Hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2"
 sequential/Hidden_layer_1/MatMul┌
0sequential/Hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0sequential/Hidden_layer_1/BiasAdd/ReadVariableOpщ
!sequential/Hidden_layer_1/BiasAddBiasAdd*sequential/Hidden_layer_1/MatMul:product:08sequential/Hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2#
!sequential/Hidden_layer_1/BiasAddп
!sequential/Hidden_layer_1/SoftmaxSoftmax*sequential/Hidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2#
!sequential/Hidden_layer_1/Softmax╒
-sequential/Output_layer/MatMul/ReadVariableOpReadVariableOp6sequential_output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02/
-sequential/Output_layer/MatMul/ReadVariableOpр
sequential/Output_layer/MatMulMatMul+sequential/Hidden_layer_1/Softmax:softmax:05sequential/Output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2 
sequential/Output_layer/MatMul╘
.sequential/Output_layer/BiasAdd/ReadVariableOpReadVariableOp7sequential_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential/Output_layer/BiasAdd/ReadVariableOpс
sequential/Output_layer/BiasAddBiasAdd(sequential/Output_layer/MatMul:product:06sequential/Output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2!
sequential/Output_layer/BiasAddй
sequential/Output_layer/SoftmaxSoftmax(sequential/Output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:         2!
sequential/Output_layer/Softmaxи
IdentityIdentity)sequential/Output_layer/Softmax:softmax:01^sequential/Hidden_layer_0/BiasAdd/ReadVariableOp0^sequential/Hidden_layer_0/MatMul/ReadVariableOp1^sequential/Hidden_layer_1/BiasAdd/ReadVariableOp0^sequential/Hidden_layer_1/MatMul/ReadVariableOp/^sequential/Output_layer/BiasAdd/ReadVariableOp.^sequential/Output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2d
0sequential/Hidden_layer_0/BiasAdd/ReadVariableOp0sequential/Hidden_layer_0/BiasAdd/ReadVariableOp2b
/sequential/Hidden_layer_0/MatMul/ReadVariableOp/sequential/Hidden_layer_0/MatMul/ReadVariableOp2d
0sequential/Hidden_layer_1/BiasAdd/ReadVariableOp0sequential/Hidden_layer_1/BiasAdd/ReadVariableOp2b
/sequential/Hidden_layer_1/MatMul/ReadVariableOp/sequential/Hidden_layer_1/MatMul/ReadVariableOp2`
.sequential/Output_layer/BiasAdd/ReadVariableOp.sequential/Output_layer/BiasAdd/ReadVariableOp2^
-sequential/Output_layer/MatMul/ReadVariableOp-sequential/Output_layer/MatMul/ReadVariableOp:T P
'
_output_shapes
:         
%
_user_specified_nameInput_layer
у
Б
,__inference_Output_layer_layer_call_fn_21185

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_Output_layer_layer_call_and_return_conditional_losses_209052
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
∙	
р
G__inference_Output_layer_layer_call_and_return_conditional_losses_21176

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ю
└
*__inference_sequential_layer_call_fn_21014
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_209992
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:         
%
_user_specified_nameInput_layer
ю
└
*__inference_sequential_layer_call_fn_20978
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_209632
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:         
%
_user_specified_nameInput_layer
╝!
■
E__inference_sequential_layer_call_and_return_conditional_losses_21066

inputs1
-hidden_layer_0_matmul_readvariableop_resource2
.hidden_layer_0_biasadd_readvariableop_resource1
-hidden_layer_1_matmul_readvariableop_resource2
.hidden_layer_1_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identityИв%Hidden_layer_0/BiasAdd/ReadVariableOpв$Hidden_layer_0/MatMul/ReadVariableOpв%Hidden_layer_1/BiasAdd/ReadVariableOpв$Hidden_layer_1/MatMul/ReadVariableOpв#Output_layer/BiasAdd/ReadVariableOpв"Output_layer/MatMul/ReadVariableOp║
$Hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$Hidden_layer_0/MatMul/ReadVariableOpа
Hidden_layer_0/MatMulMatMulinputs,Hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Hidden_layer_0/MatMul╣
%Hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Hidden_layer_0/BiasAdd/ReadVariableOp╜
Hidden_layer_0/BiasAddBiasAddHidden_layer_0/MatMul:product:0-Hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Hidden_layer_0/BiasAddО
Hidden_layer_0/SoftmaxSoftmaxHidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:         2
Hidden_layer_0/Softmax║
$Hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$Hidden_layer_1/MatMul/ReadVariableOp║
Hidden_layer_1/MatMulMatMul Hidden_layer_0/Softmax:softmax:0,Hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Hidden_layer_1/MatMul╣
%Hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Hidden_layer_1/BiasAdd/ReadVariableOp╜
Hidden_layer_1/BiasAddBiasAddHidden_layer_1/MatMul:product:0-Hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Hidden_layer_1/BiasAddО
Hidden_layer_1/SoftmaxSoftmaxHidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
Hidden_layer_1/Softmax┤
"Output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"Output_layer/MatMul/ReadVariableOp┤
Output_layer/MatMulMatMul Hidden_layer_1/Softmax:softmax:0*Output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Output_layer/MatMul│
#Output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#Output_layer/BiasAdd/ReadVariableOp╡
Output_layer/BiasAddBiasAddOutput_layer/MatMul:product:0+Output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Output_layer/BiasAddИ
Output_layer/SoftmaxSoftmaxOutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:         2
Output_layer/Softmax█
IdentityIdentityOutput_layer/Softmax:softmax:0&^Hidden_layer_0/BiasAdd/ReadVariableOp%^Hidden_layer_0/MatMul/ReadVariableOp&^Hidden_layer_1/BiasAdd/ReadVariableOp%^Hidden_layer_1/MatMul/ReadVariableOp$^Output_layer/BiasAdd/ReadVariableOp#^Output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2N
%Hidden_layer_0/BiasAdd/ReadVariableOp%Hidden_layer_0/BiasAdd/ReadVariableOp2L
$Hidden_layer_0/MatMul/ReadVariableOp$Hidden_layer_0/MatMul/ReadVariableOp2N
%Hidden_layer_1/BiasAdd/ReadVariableOp%Hidden_layer_1/BiasAdd/ReadVariableOp2L
$Hidden_layer_1/MatMul/ReadVariableOp$Hidden_layer_1/MatMul/ReadVariableOp2J
#Output_layer/BiasAdd/ReadVariableOp#Output_layer/BiasAdd/ReadVariableOp2H
"Output_layer/MatMul/ReadVariableOp"Output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
│t
├
!__inference__traced_restore_21380
file_prefix*
&assignvariableop_hidden_layer_0_kernel*
&assignvariableop_1_hidden_layer_0_bias,
(assignvariableop_2_hidden_layer_1_kernel*
&assignvariableop_3_hidden_layer_1_bias*
&assignvariableop_4_output_layer_kernel(
$assignvariableop_5_output_layer_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count
assignvariableop_13_total_1
assignvariableop_14_count_14
0assignvariableop_15_adam_hidden_layer_0_kernel_m2
.assignvariableop_16_adam_hidden_layer_0_bias_m4
0assignvariableop_17_adam_hidden_layer_1_kernel_m2
.assignvariableop_18_adam_hidden_layer_1_bias_m2
.assignvariableop_19_adam_output_layer_kernel_m0
,assignvariableop_20_adam_output_layer_bias_m4
0assignvariableop_21_adam_hidden_layer_0_kernel_v2
.assignvariableop_22_adam_hidden_layer_0_bias_v4
0assignvariableop_23_adam_hidden_layer_1_kernel_v2
.assignvariableop_24_adam_hidden_layer_1_bias_v2
.assignvariableop_25_adam_output_layer_kernel_v0
,assignvariableop_26_adam_output_layer_bias_v
identity_28ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9Т
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ю
valueФBСB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names╞
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices╕
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Д
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityе
AssignVariableOpAssignVariableOp&assignvariableop_hidden_layer_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1л
AssignVariableOp_1AssignVariableOp&assignvariableop_1_hidden_layer_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2н
AssignVariableOp_2AssignVariableOp(assignvariableop_2_hidden_layer_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3л
AssignVariableOp_3AssignVariableOp&assignvariableop_3_hidden_layer_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4л
AssignVariableOp_4AssignVariableOp&assignvariableop_4_output_layer_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5й
AssignVariableOp_5AssignVariableOp$assignvariableop_5_output_layer_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6б
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7г
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8г
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9в
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10о
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11б
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12б
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13г
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14г
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15╕
AssignVariableOp_15AssignVariableOp0assignvariableop_15_adam_hidden_layer_0_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16╢
AssignVariableOp_16AssignVariableOp.assignvariableop_16_adam_hidden_layer_0_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17╕
AssignVariableOp_17AssignVariableOp0assignvariableop_17_adam_hidden_layer_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18╢
AssignVariableOp_18AssignVariableOp.assignvariableop_18_adam_hidden_layer_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19╢
AssignVariableOp_19AssignVariableOp.assignvariableop_19_adam_output_layer_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20┤
AssignVariableOp_20AssignVariableOp,assignvariableop_20_adam_output_layer_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21╕
AssignVariableOp_21AssignVariableOp0assignvariableop_21_adam_hidden_layer_0_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22╢
AssignVariableOp_22AssignVariableOp.assignvariableop_22_adam_hidden_layer_0_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23╕
AssignVariableOp_23AssignVariableOp0assignvariableop_23_adam_hidden_layer_1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24╢
AssignVariableOp_24AssignVariableOp.assignvariableop_24_adam_hidden_layer_1_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25╢
AssignVariableOp_25AssignVariableOp.assignvariableop_25_adam_output_layer_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26┤
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_output_layer_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_269
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp░
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_27г
Identity_28IdentityIdentity_27:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_28"#
identity_28Identity_28:output:0*Б
_input_shapesp
n: :::::::::::::::::::::::::::2$
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
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ї
ї
E__inference_sequential_layer_call_and_return_conditional_losses_20999

inputs
hidden_layer_0_20983
hidden_layer_0_20985
hidden_layer_1_20988
hidden_layer_1_20990
output_layer_20993
output_layer_20995
identityИв&Hidden_layer_0/StatefulPartitionedCallв&Hidden_layer_1/StatefulPartitionedCallв$Output_layer/StatefulPartitionedCallп
&Hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinputshidden_layer_0_20983hidden_layer_0_20985*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_208512(
&Hidden_layer_0/StatefulPartitionedCall╪
&Hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_20988hidden_layer_1_20990*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_208782(
&Hidden_layer_1/StatefulPartitionedCall╬
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_1/StatefulPartitionedCall:output:0output_layer_20993output_layer_20995*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_Output_layer_layer_call_and_return_conditional_losses_209052&
$Output_layer/StatefulPartitionedCall·
IdentityIdentity-Output_layer/StatefulPartitionedCall:output:0'^Hidden_layer_0/StatefulPartitionedCall'^Hidden_layer_1/StatefulPartitionedCall%^Output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2P
&Hidden_layer_0/StatefulPartitionedCall&Hidden_layer_0/StatefulPartitionedCall2P
&Hidden_layer_1/StatefulPartitionedCall&Hidden_layer_1/StatefulPartitionedCall2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
√	
т
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_20851

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┬
╣
#__inference_signature_wrapper_21041
input_layer
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_208362
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:         
%
_user_specified_nameInput_layer
Д
·
E__inference_sequential_layer_call_and_return_conditional_losses_20922
input_layer
hidden_layer_0_20862
hidden_layer_0_20864
hidden_layer_1_20889
hidden_layer_1_20891
output_layer_20916
output_layer_20918
identityИв&Hidden_layer_0/StatefulPartitionedCallв&Hidden_layer_1/StatefulPartitionedCallв$Output_layer/StatefulPartitionedCall┤
&Hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinput_layerhidden_layer_0_20862hidden_layer_0_20864*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_208512(
&Hidden_layer_0/StatefulPartitionedCall╪
&Hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_0/StatefulPartitionedCall:output:0hidden_layer_1_20889hidden_layer_1_20891*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_208782(
&Hidden_layer_1/StatefulPartitionedCall╬
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden_layer_1/StatefulPartitionedCall:output:0output_layer_20916output_layer_20918*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_Output_layer_layer_call_and_return_conditional_losses_209052&
$Output_layer/StatefulPartitionedCall·
IdentityIdentity-Output_layer/StatefulPartitionedCall:output:0'^Hidden_layer_0/StatefulPartitionedCall'^Hidden_layer_1/StatefulPartitionedCall%^Output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2P
&Hidden_layer_0/StatefulPartitionedCall&Hidden_layer_0/StatefulPartitionedCall2P
&Hidden_layer_1/StatefulPartitionedCall&Hidden_layer_1/StatefulPartitionedCall2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall:T P
'
_output_shapes
:         
%
_user_specified_nameInput_layer
√	
т
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_21156

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
▀>
щ
__inference__traced_save_21289
file_prefix4
0savev2_hidden_layer_0_kernel_read_readvariableop2
.savev2_hidden_layer_0_bias_read_readvariableop4
0savev2_hidden_layer_1_kernel_read_readvariableop2
.savev2_hidden_layer_1_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop;
7savev2_adam_hidden_layer_0_kernel_m_read_readvariableop9
5savev2_adam_hidden_layer_0_bias_m_read_readvariableop;
7savev2_adam_hidden_layer_1_kernel_m_read_readvariableop9
5savev2_adam_hidden_layer_1_bias_m_read_readvariableop9
5savev2_adam_output_layer_kernel_m_read_readvariableop7
3savev2_adam_output_layer_bias_m_read_readvariableop;
7savev2_adam_hidden_layer_0_kernel_v_read_readvariableop9
5savev2_adam_hidden_layer_0_bias_v_read_readvariableop;
7savev2_adam_hidden_layer_1_kernel_v_read_readvariableop9
5savev2_adam_hidden_layer_1_bias_v_read_readvariableop9
5savev2_adam_output_layer_kernel_v_read_readvariableop7
3savev2_adam_output_layer_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameМ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ю
valueФBСB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names└
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesч
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_hidden_layer_0_kernel_read_readvariableop.savev2_hidden_layer_0_bias_read_readvariableop0savev2_hidden_layer_1_kernel_read_readvariableop.savev2_hidden_layer_1_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop7savev2_adam_hidden_layer_0_kernel_m_read_readvariableop5savev2_adam_hidden_layer_0_bias_m_read_readvariableop7savev2_adam_hidden_layer_1_kernel_m_read_readvariableop5savev2_adam_hidden_layer_1_bias_m_read_readvariableop5savev2_adam_output_layer_kernel_m_read_readvariableop3savev2_adam_output_layer_bias_m_read_readvariableop7savev2_adam_hidden_layer_0_kernel_v_read_readvariableop5savev2_adam_hidden_layer_0_bias_v_read_readvariableop7savev2_adam_hidden_layer_1_kernel_v_read_readvariableop5savev2_adam_hidden_layer_1_bias_v_read_readvariableop5savev2_adam_output_layer_kernel_v_read_readvariableop3savev2_adam_output_layer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*╗
_input_shapesй
ж: ::::::: : : : : : : : : ::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
▀
╗
*__inference_sequential_layer_call_fn_21108

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_209632
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ч
Г
.__inference_Hidden_layer_0_layer_call_fn_21145

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_208512
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╖
serving_defaultг
C
Input_layer4
serving_default_Input_layer:0         @
Output_layer0
StatefulPartitionedCall:0         tensorflow/serving/predict:ЙЗ
╚#
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
L_default_save_signature
M__call__
*N&call_and_return_all_conditional_losses"И!
_tf_keras_sequentialщ {"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input_layer"}}, {"class_name": "Dense", "config": {"name": "Hidden_layer_0", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Hidden_layer_1", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Output_layer", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input_layer"}}, {"class_name": "Dense", "config": {"name": "Hidden_layer_0", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Hidden_layer_1", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Output_layer", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": false}}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
А


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
O__call__
*P&call_and_return_all_conditional_losses"█
_tf_keras_layer┴{"class_name": "Dense", "name": "Hidden_layer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Hidden_layer_0", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12]}}
■

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"┘
_tf_keras_layer┐{"class_name": "Dense", "name": "Hidden_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Hidden_layer_1", "trainable": true, "dtype": "float32", "units": 6, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
·

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
S__call__
*T&call_and_return_all_conditional_losses"╒
_tf_keras_layer╗{"class_name": "Dense", "name": "Output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Output_layer", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
┐
iter

beta_1

beta_2
	decay
 learning_rate
m@mAmBmCmDmE
vFvGvHvIvJvK"
	optimizer
 "
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
╩
regularization_losses
!layer_metrics
	variables

"layers
#non_trainable_variables
$metrics
%layer_regularization_losses
trainable_variables
M__call__
L_default_save_signature
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
,
Userving_default"
signature_map
':%2Hidden_layer_0/kernel
!:2Hidden_layer_0/bias
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
н
regularization_losses
&layer_metrics
'non_trainable_variables

(layers
	variables
)metrics
*layer_regularization_losses
trainable_variables
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
':%2Hidden_layer_1/kernel
!:2Hidden_layer_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
regularization_losses
+layer_metrics
,non_trainable_variables

-layers
	variables
.metrics
/layer_regularization_losses
trainable_variables
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
%:#2Output_layer/kernel
:2Output_layer/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
regularization_losses
0layer_metrics
1non_trainable_variables

2layers
	variables
3metrics
4layer_regularization_losses
trainable_variables
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╗
	7total
	8count
9	variables
:	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
м
	;total
	<count
=
_fn_kwargs
>	variables
?	keras_api"х
_tf_keras_metric╩{"class_name": "MeanMetricWrapper", "name": "sparse_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
.
70
81"
trackable_list_wrapper
-
9	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
;0
<1"
trackable_list_wrapper
-
>	variables"
_generic_user_object
,:*2Adam/Hidden_layer_0/kernel/m
&:$2Adam/Hidden_layer_0/bias/m
,:*2Adam/Hidden_layer_1/kernel/m
&:$2Adam/Hidden_layer_1/bias/m
*:(2Adam/Output_layer/kernel/m
$:"2Adam/Output_layer/bias/m
,:*2Adam/Hidden_layer_0/kernel/v
&:$2Adam/Hidden_layer_0/bias/v
,:*2Adam/Hidden_layer_1/kernel/v
&:$2Adam/Hidden_layer_1/bias/v
*:(2Adam/Output_layer/kernel/v
$:"2Adam/Output_layer/bias/v
т2▀
 __inference__wrapped_model_20836║
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк **в'
%К"
Input_layer         
Ў2є
*__inference_sequential_layer_call_fn_21014
*__inference_sequential_layer_call_fn_21125
*__inference_sequential_layer_call_fn_21108
*__inference_sequential_layer_call_fn_20978└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
т2▀
E__inference_sequential_layer_call_and_return_conditional_losses_20922
E__inference_sequential_layer_call_and_return_conditional_losses_21066
E__inference_sequential_layer_call_and_return_conditional_losses_21091
E__inference_sequential_layer_call_and_return_conditional_losses_20941└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╪2╒
.__inference_Hidden_layer_0_layer_call_fn_21145в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
є2Ё
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_21136в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╪2╒
.__inference_Hidden_layer_1_layer_call_fn_21165в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
є2Ё
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_21156в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_Output_layer_layer_call_fn_21185в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_Output_layer_layer_call_and_return_conditional_losses_21176в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╬B╦
#__inference_signature_wrapper_21041Input_layer"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 й
I__inference_Hidden_layer_0_layer_call_and_return_conditional_losses_21136\
/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ Б
.__inference_Hidden_layer_0_layer_call_fn_21145O
/в,
%в"
 К
inputs         
к "К         й
I__inference_Hidden_layer_1_layer_call_and_return_conditional_losses_21156\/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ Б
.__inference_Hidden_layer_1_layer_call_fn_21165O/в,
%в"
 К
inputs         
к "К         з
G__inference_Output_layer_layer_call_and_return_conditional_losses_21176\/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ 
,__inference_Output_layer_layer_call_fn_21185O/в,
%в"
 К
inputs         
к "К         Я
 __inference__wrapped_model_20836{
4в1
*в'
%К"
Input_layer         
к ";к8
6
Output_layer&К#
Output_layer         ╢
E__inference_sequential_layer_call_and_return_conditional_losses_20922m
<в9
2в/
%К"
Input_layer         
p

 
к "%в"
К
0         
Ъ ╢
E__inference_sequential_layer_call_and_return_conditional_losses_20941m
<в9
2в/
%К"
Input_layer         
p 

 
к "%в"
К
0         
Ъ ▒
E__inference_sequential_layer_call_and_return_conditional_losses_21066h
7в4
-в*
 К
inputs         
p

 
к "%в"
К
0         
Ъ ▒
E__inference_sequential_layer_call_and_return_conditional_losses_21091h
7в4
-в*
 К
inputs         
p 

 
к "%в"
К
0         
Ъ О
*__inference_sequential_layer_call_fn_20978`
<в9
2в/
%К"
Input_layer         
p

 
к "К         О
*__inference_sequential_layer_call_fn_21014`
<в9
2в/
%К"
Input_layer         
p 

 
к "К         Й
*__inference_sequential_layer_call_fn_21108[
7в4
-в*
 К
inputs         
p

 
к "К         Й
*__inference_sequential_layer_call_fn_21125[
7в4
-в*
 К
inputs         
p 

 
к "К         ▓
#__inference_signature_wrapper_21041К
Cв@
в 
9к6
4
Input_layer%К"
Input_layer         ";к8
6
Output_layer&К#
Output_layer         