��:
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48��0
�
random_zoom_2/StateVarVarHandleOp*
_output_shapes
: *'

debug_namerandom_zoom_2/StateVar/*
dtype0	*
shape:*'
shared_namerandom_zoom_2/StateVar
}
*random_zoom_2/StateVar/Read/ReadVariableOpReadVariableOprandom_zoom_2/StateVar*
_output_shapes
:*
dtype0	
�
random_rotation_2/StateVarVarHandleOp*
_output_shapes
: *+

debug_namerandom_rotation_2/StateVar/*
dtype0	*
shape:*+
shared_namerandom_rotation_2/StateVar
�
.random_rotation_2/StateVar/Read/ReadVariableOpReadVariableOprandom_rotation_2/StateVar*
_output_shapes
:*
dtype0	
�
random_flip_2/StateVarVarHandleOp*
_output_shapes
: *'

debug_namerandom_flip_2/StateVar/*
dtype0	*
shape:*'
shared_namerandom_flip_2/StateVar
}
*random_flip_2/StateVar/Read/ReadVariableOpReadVariableOprandom_flip_2/StateVar*
_output_shapes
:*
dtype0	
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
RMSprop/velocity/dense_2/biasVarHandleOp*
_output_shapes
: *.

debug_name RMSprop/velocity/dense_2/bias/*
dtype0*
shape:*.
shared_nameRMSprop/velocity/dense_2/bias
�
1RMSprop/velocity/dense_2/bias/Read/ReadVariableOpReadVariableOpRMSprop/velocity/dense_2/bias*
_output_shapes
:*
dtype0
�
RMSprop/velocity/dense_2/kernelVarHandleOp*
_output_shapes
: *0

debug_name" RMSprop/velocity/dense_2/kernel/*
dtype0*
shape:	�*0
shared_name!RMSprop/velocity/dense_2/kernel
�
3RMSprop/velocity/dense_2/kernel/Read/ReadVariableOpReadVariableOpRMSprop/velocity/dense_2/kernel*
_output_shapes
:	�*
dtype0
�
!RMSprop/velocity/conv2d_15/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"RMSprop/velocity/conv2d_15/kernel/*
dtype0*
shape:��*2
shared_name#!RMSprop/velocity/conv2d_15/kernel
�
5RMSprop/velocity/conv2d_15/kernel/Read/ReadVariableOpReadVariableOp!RMSprop/velocity/conv2d_15/kernel*(
_output_shapes
:��*
dtype0
�
4RMSprop/velocity/separable_conv2d_8/pointwise_kernelVarHandleOp*
_output_shapes
: *E

debug_name75RMSprop/velocity/separable_conv2d_8/pointwise_kernel/*
dtype0*
shape:��*E
shared_name64RMSprop/velocity/separable_conv2d_8/pointwise_kernel
�
HRMSprop/velocity/separable_conv2d_8/pointwise_kernel/Read/ReadVariableOpReadVariableOp4RMSprop/velocity/separable_conv2d_8/pointwise_kernel*(
_output_shapes
:��*
dtype0
�
4RMSprop/velocity/separable_conv2d_8/depthwise_kernelVarHandleOp*
_output_shapes
: *E

debug_name75RMSprop/velocity/separable_conv2d_8/depthwise_kernel/*
dtype0*
shape:�*E
shared_name64RMSprop/velocity/separable_conv2d_8/depthwise_kernel
�
HRMSprop/velocity/separable_conv2d_8/depthwise_kernel/Read/ReadVariableOpReadVariableOp4RMSprop/velocity/separable_conv2d_8/depthwise_kernel*'
_output_shapes
:�*
dtype0
�
+RMSprop/velocity/batch_normalization_8/betaVarHandleOp*
_output_shapes
: *<

debug_name.,RMSprop/velocity/batch_normalization_8/beta/*
dtype0*
shape:�*<
shared_name-+RMSprop/velocity/batch_normalization_8/beta
�
?RMSprop/velocity/batch_normalization_8/beta/Read/ReadVariableOpReadVariableOp+RMSprop/velocity/batch_normalization_8/beta*
_output_shapes	
:�*
dtype0
�
,RMSprop/velocity/batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *=

debug_name/-RMSprop/velocity/batch_normalization_8/gamma/*
dtype0*
shape:�*=
shared_name.,RMSprop/velocity/batch_normalization_8/gamma
�
@RMSprop/velocity/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOp,RMSprop/velocity/batch_normalization_8/gamma*
_output_shapes	
:�*
dtype0
�
4RMSprop/velocity/separable_conv2d_9/pointwise_kernelVarHandleOp*
_output_shapes
: *E

debug_name75RMSprop/velocity/separable_conv2d_9/pointwise_kernel/*
dtype0*
shape:��*E
shared_name64RMSprop/velocity/separable_conv2d_9/pointwise_kernel
�
HRMSprop/velocity/separable_conv2d_9/pointwise_kernel/Read/ReadVariableOpReadVariableOp4RMSprop/velocity/separable_conv2d_9/pointwise_kernel*(
_output_shapes
:��*
dtype0
�
4RMSprop/velocity/separable_conv2d_9/depthwise_kernelVarHandleOp*
_output_shapes
: *E

debug_name75RMSprop/velocity/separable_conv2d_9/depthwise_kernel/*
dtype0*
shape:�*E
shared_name64RMSprop/velocity/separable_conv2d_9/depthwise_kernel
�
HRMSprop/velocity/separable_conv2d_9/depthwise_kernel/Read/ReadVariableOpReadVariableOp4RMSprop/velocity/separable_conv2d_9/depthwise_kernel*'
_output_shapes
:�*
dtype0
�
+RMSprop/velocity/batch_normalization_9/betaVarHandleOp*
_output_shapes
: *<

debug_name.,RMSprop/velocity/batch_normalization_9/beta/*
dtype0*
shape:�*<
shared_name-+RMSprop/velocity/batch_normalization_9/beta
�
?RMSprop/velocity/batch_normalization_9/beta/Read/ReadVariableOpReadVariableOp+RMSprop/velocity/batch_normalization_9/beta*
_output_shapes	
:�*
dtype0
�
,RMSprop/velocity/batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *=

debug_name/-RMSprop/velocity/batch_normalization_9/gamma/*
dtype0*
shape:�*=
shared_name.,RMSprop/velocity/batch_normalization_9/gamma
�
@RMSprop/velocity/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOp,RMSprop/velocity/batch_normalization_9/gamma*
_output_shapes	
:�*
dtype0
�
!RMSprop/velocity/conv2d_14/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"RMSprop/velocity/conv2d_14/kernel/*
dtype0*
shape:��*2
shared_name#!RMSprop/velocity/conv2d_14/kernel
�
5RMSprop/velocity/conv2d_14/kernel/Read/ReadVariableOpReadVariableOp!RMSprop/velocity/conv2d_14/kernel*(
_output_shapes
:��*
dtype0
�
4RMSprop/velocity/separable_conv2d_6/pointwise_kernelVarHandleOp*
_output_shapes
: *E

debug_name75RMSprop/velocity/separable_conv2d_6/pointwise_kernel/*
dtype0*
shape:��*E
shared_name64RMSprop/velocity/separable_conv2d_6/pointwise_kernel
�
HRMSprop/velocity/separable_conv2d_6/pointwise_kernel/Read/ReadVariableOpReadVariableOp4RMSprop/velocity/separable_conv2d_6/pointwise_kernel*(
_output_shapes
:��*
dtype0
�
4RMSprop/velocity/separable_conv2d_6/depthwise_kernelVarHandleOp*
_output_shapes
: *E

debug_name75RMSprop/velocity/separable_conv2d_6/depthwise_kernel/*
dtype0*
shape:�*E
shared_name64RMSprop/velocity/separable_conv2d_6/depthwise_kernel
�
HRMSprop/velocity/separable_conv2d_6/depthwise_kernel/Read/ReadVariableOpReadVariableOp4RMSprop/velocity/separable_conv2d_6/depthwise_kernel*'
_output_shapes
:�*
dtype0
�
+RMSprop/velocity/batch_normalization_6/betaVarHandleOp*
_output_shapes
: *<

debug_name.,RMSprop/velocity/batch_normalization_6/beta/*
dtype0*
shape:�*<
shared_name-+RMSprop/velocity/batch_normalization_6/beta
�
?RMSprop/velocity/batch_normalization_6/beta/Read/ReadVariableOpReadVariableOp+RMSprop/velocity/batch_normalization_6/beta*
_output_shapes	
:�*
dtype0
�
,RMSprop/velocity/batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *=

debug_name/-RMSprop/velocity/batch_normalization_6/gamma/*
dtype0*
shape:�*=
shared_name.,RMSprop/velocity/batch_normalization_6/gamma
�
@RMSprop/velocity/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOp,RMSprop/velocity/batch_normalization_6/gamma*
_output_shapes	
:�*
dtype0
�
4RMSprop/velocity/separable_conv2d_7/pointwise_kernelVarHandleOp*
_output_shapes
: *E

debug_name75RMSprop/velocity/separable_conv2d_7/pointwise_kernel/*
dtype0*
shape:��*E
shared_name64RMSprop/velocity/separable_conv2d_7/pointwise_kernel
�
HRMSprop/velocity/separable_conv2d_7/pointwise_kernel/Read/ReadVariableOpReadVariableOp4RMSprop/velocity/separable_conv2d_7/pointwise_kernel*(
_output_shapes
:��*
dtype0
�
4RMSprop/velocity/separable_conv2d_7/depthwise_kernelVarHandleOp*
_output_shapes
: *E

debug_name75RMSprop/velocity/separable_conv2d_7/depthwise_kernel/*
dtype0*
shape:�*E
shared_name64RMSprop/velocity/separable_conv2d_7/depthwise_kernel
�
HRMSprop/velocity/separable_conv2d_7/depthwise_kernel/Read/ReadVariableOpReadVariableOp4RMSprop/velocity/separable_conv2d_7/depthwise_kernel*'
_output_shapes
:�*
dtype0
�
+RMSprop/velocity/batch_normalization_7/betaVarHandleOp*
_output_shapes
: *<

debug_name.,RMSprop/velocity/batch_normalization_7/beta/*
dtype0*
shape:�*<
shared_name-+RMSprop/velocity/batch_normalization_7/beta
�
?RMSprop/velocity/batch_normalization_7/beta/Read/ReadVariableOpReadVariableOp+RMSprop/velocity/batch_normalization_7/beta*
_output_shapes	
:�*
dtype0
�
,RMSprop/velocity/batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *=

debug_name/-RMSprop/velocity/batch_normalization_7/gamma/*
dtype0*
shape:�*=
shared_name.,RMSprop/velocity/batch_normalization_7/gamma
�
@RMSprop/velocity/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOp,RMSprop/velocity/batch_normalization_7/gamma*
_output_shapes	
:�*
dtype0
�
!RMSprop/velocity/conv2d_13/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"RMSprop/velocity/conv2d_13/kernel/*
dtype0*
shape:@�*2
shared_name#!RMSprop/velocity/conv2d_13/kernel
�
5RMSprop/velocity/conv2d_13/kernel/Read/ReadVariableOpReadVariableOp!RMSprop/velocity/conv2d_13/kernel*'
_output_shapes
:@�*
dtype0
�
4RMSprop/velocity/separable_conv2d_4/pointwise_kernelVarHandleOp*
_output_shapes
: *E

debug_name75RMSprop/velocity/separable_conv2d_4/pointwise_kernel/*
dtype0*
shape:��*E
shared_name64RMSprop/velocity/separable_conv2d_4/pointwise_kernel
�
HRMSprop/velocity/separable_conv2d_4/pointwise_kernel/Read/ReadVariableOpReadVariableOp4RMSprop/velocity/separable_conv2d_4/pointwise_kernel*(
_output_shapes
:��*
dtype0
�
4RMSprop/velocity/separable_conv2d_4/depthwise_kernelVarHandleOp*
_output_shapes
: *E

debug_name75RMSprop/velocity/separable_conv2d_4/depthwise_kernel/*
dtype0*
shape:�*E
shared_name64RMSprop/velocity/separable_conv2d_4/depthwise_kernel
�
HRMSprop/velocity/separable_conv2d_4/depthwise_kernel/Read/ReadVariableOpReadVariableOp4RMSprop/velocity/separable_conv2d_4/depthwise_kernel*'
_output_shapes
:�*
dtype0
�
+RMSprop/velocity/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *<

debug_name.,RMSprop/velocity/batch_normalization_4/beta/*
dtype0*
shape:�*<
shared_name-+RMSprop/velocity/batch_normalization_4/beta
�
?RMSprop/velocity/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp+RMSprop/velocity/batch_normalization_4/beta*
_output_shapes	
:�*
dtype0
�
,RMSprop/velocity/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *=

debug_name/-RMSprop/velocity/batch_normalization_4/gamma/*
dtype0*
shape:�*=
shared_name.,RMSprop/velocity/batch_normalization_4/gamma
�
@RMSprop/velocity/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp,RMSprop/velocity/batch_normalization_4/gamma*
_output_shapes	
:�*
dtype0
�
4RMSprop/velocity/separable_conv2d_5/pointwise_kernelVarHandleOp*
_output_shapes
: *E

debug_name75RMSprop/velocity/separable_conv2d_5/pointwise_kernel/*
dtype0*
shape:@�*E
shared_name64RMSprop/velocity/separable_conv2d_5/pointwise_kernel
�
HRMSprop/velocity/separable_conv2d_5/pointwise_kernel/Read/ReadVariableOpReadVariableOp4RMSprop/velocity/separable_conv2d_5/pointwise_kernel*'
_output_shapes
:@�*
dtype0
�
4RMSprop/velocity/separable_conv2d_5/depthwise_kernelVarHandleOp*
_output_shapes
: *E

debug_name75RMSprop/velocity/separable_conv2d_5/depthwise_kernel/*
dtype0*
shape:@*E
shared_name64RMSprop/velocity/separable_conv2d_5/depthwise_kernel
�
HRMSprop/velocity/separable_conv2d_5/depthwise_kernel/Read/ReadVariableOpReadVariableOp4RMSprop/velocity/separable_conv2d_5/depthwise_kernel*&
_output_shapes
:@*
dtype0
�
+RMSprop/velocity/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *<

debug_name.,RMSprop/velocity/batch_normalization_5/beta/*
dtype0*
shape:@*<
shared_name-+RMSprop/velocity/batch_normalization_5/beta
�
?RMSprop/velocity/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp+RMSprop/velocity/batch_normalization_5/beta*
_output_shapes
:@*
dtype0
�
,RMSprop/velocity/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *=

debug_name/-RMSprop/velocity/batch_normalization_5/gamma/*
dtype0*
shape:@*=
shared_name.,RMSprop/velocity/batch_normalization_5/gamma
�
@RMSprop/velocity/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp,RMSprop/velocity/batch_normalization_5/gamma*
_output_shapes
:@*
dtype0
�
!RMSprop/velocity/conv2d_12/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"RMSprop/velocity/conv2d_12/kernel/*
dtype0*
shape: @*2
shared_name#!RMSprop/velocity/conv2d_12/kernel
�
5RMSprop/velocity/conv2d_12/kernel/Read/ReadVariableOpReadVariableOp!RMSprop/velocity/conv2d_12/kernel*&
_output_shapes
: @*
dtype0
�
4RMSprop/velocity/separable_conv2d_2/pointwise_kernelVarHandleOp*
_output_shapes
: *E

debug_name75RMSprop/velocity/separable_conv2d_2/pointwise_kernel/*
dtype0*
shape:@@*E
shared_name64RMSprop/velocity/separable_conv2d_2/pointwise_kernel
�
HRMSprop/velocity/separable_conv2d_2/pointwise_kernel/Read/ReadVariableOpReadVariableOp4RMSprop/velocity/separable_conv2d_2/pointwise_kernel*&
_output_shapes
:@@*
dtype0
�
4RMSprop/velocity/separable_conv2d_2/depthwise_kernelVarHandleOp*
_output_shapes
: *E

debug_name75RMSprop/velocity/separable_conv2d_2/depthwise_kernel/*
dtype0*
shape:@*E
shared_name64RMSprop/velocity/separable_conv2d_2/depthwise_kernel
�
HRMSprop/velocity/separable_conv2d_2/depthwise_kernel/Read/ReadVariableOpReadVariableOp4RMSprop/velocity/separable_conv2d_2/depthwise_kernel*&
_output_shapes
:@*
dtype0
�
+RMSprop/velocity/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *<

debug_name.,RMSprop/velocity/batch_normalization_2/beta/*
dtype0*
shape:@*<
shared_name-+RMSprop/velocity/batch_normalization_2/beta
�
?RMSprop/velocity/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp+RMSprop/velocity/batch_normalization_2/beta*
_output_shapes
:@*
dtype0
�
,RMSprop/velocity/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *=

debug_name/-RMSprop/velocity/batch_normalization_2/gamma/*
dtype0*
shape:@*=
shared_name.,RMSprop/velocity/batch_normalization_2/gamma
�
@RMSprop/velocity/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp,RMSprop/velocity/batch_normalization_2/gamma*
_output_shapes
:@*
dtype0
�
4RMSprop/velocity/separable_conv2d_3/pointwise_kernelVarHandleOp*
_output_shapes
: *E

debug_name75RMSprop/velocity/separable_conv2d_3/pointwise_kernel/*
dtype0*
shape: @*E
shared_name64RMSprop/velocity/separable_conv2d_3/pointwise_kernel
�
HRMSprop/velocity/separable_conv2d_3/pointwise_kernel/Read/ReadVariableOpReadVariableOp4RMSprop/velocity/separable_conv2d_3/pointwise_kernel*&
_output_shapes
: @*
dtype0
�
4RMSprop/velocity/separable_conv2d_3/depthwise_kernelVarHandleOp*
_output_shapes
: *E

debug_name75RMSprop/velocity/separable_conv2d_3/depthwise_kernel/*
dtype0*
shape: *E
shared_name64RMSprop/velocity/separable_conv2d_3/depthwise_kernel
�
HRMSprop/velocity/separable_conv2d_3/depthwise_kernel/Read/ReadVariableOpReadVariableOp4RMSprop/velocity/separable_conv2d_3/depthwise_kernel*&
_output_shapes
: *
dtype0
�
+RMSprop/velocity/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *<

debug_name.,RMSprop/velocity/batch_normalization_3/beta/*
dtype0*
shape: *<
shared_name-+RMSprop/velocity/batch_normalization_3/beta
�
?RMSprop/velocity/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp+RMSprop/velocity/batch_normalization_3/beta*
_output_shapes
: *
dtype0
�
,RMSprop/velocity/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *=

debug_name/-RMSprop/velocity/batch_normalization_3/gamma/*
dtype0*
shape: *=
shared_name.,RMSprop/velocity/batch_normalization_3/gamma
�
@RMSprop/velocity/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp,RMSprop/velocity/batch_normalization_3/gamma*
_output_shapes
: *
dtype0
�
!RMSprop/velocity/conv2d_11/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"RMSprop/velocity/conv2d_11/kernel/*
dtype0*
shape:  *2
shared_name#!RMSprop/velocity/conv2d_11/kernel
�
5RMSprop/velocity/conv2d_11/kernel/Read/ReadVariableOpReadVariableOp!RMSprop/velocity/conv2d_11/kernel*&
_output_shapes
:  *
dtype0
�
2RMSprop/velocity/separable_conv2d/pointwise_kernelVarHandleOp*
_output_shapes
: *C

debug_name53RMSprop/velocity/separable_conv2d/pointwise_kernel/*
dtype0*
shape:  *C
shared_name42RMSprop/velocity/separable_conv2d/pointwise_kernel
�
FRMSprop/velocity/separable_conv2d/pointwise_kernel/Read/ReadVariableOpReadVariableOp2RMSprop/velocity/separable_conv2d/pointwise_kernel*&
_output_shapes
:  *
dtype0
�
2RMSprop/velocity/separable_conv2d/depthwise_kernelVarHandleOp*
_output_shapes
: *C

debug_name53RMSprop/velocity/separable_conv2d/depthwise_kernel/*
dtype0*
shape: *C
shared_name42RMSprop/velocity/separable_conv2d/depthwise_kernel
�
FRMSprop/velocity/separable_conv2d/depthwise_kernel/Read/ReadVariableOpReadVariableOp2RMSprop/velocity/separable_conv2d/depthwise_kernel*&
_output_shapes
: *
dtype0
�
)RMSprop/velocity/batch_normalization/betaVarHandleOp*
_output_shapes
: *:

debug_name,*RMSprop/velocity/batch_normalization/beta/*
dtype0*
shape: *:
shared_name+)RMSprop/velocity/batch_normalization/beta
�
=RMSprop/velocity/batch_normalization/beta/Read/ReadVariableOpReadVariableOp)RMSprop/velocity/batch_normalization/beta*
_output_shapes
: *
dtype0
�
*RMSprop/velocity/batch_normalization/gammaVarHandleOp*
_output_shapes
: *;

debug_name-+RMSprop/velocity/batch_normalization/gamma/*
dtype0*
shape: *;
shared_name,*RMSprop/velocity/batch_normalization/gamma
�
>RMSprop/velocity/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp*RMSprop/velocity/batch_normalization/gamma*
_output_shapes
: *
dtype0
�
4RMSprop/velocity/separable_conv2d_1/pointwise_kernelVarHandleOp*
_output_shapes
: *E

debug_name75RMSprop/velocity/separable_conv2d_1/pointwise_kernel/*
dtype0*
shape:  *E
shared_name64RMSprop/velocity/separable_conv2d_1/pointwise_kernel
�
HRMSprop/velocity/separable_conv2d_1/pointwise_kernel/Read/ReadVariableOpReadVariableOp4RMSprop/velocity/separable_conv2d_1/pointwise_kernel*&
_output_shapes
:  *
dtype0
�
4RMSprop/velocity/separable_conv2d_1/depthwise_kernelVarHandleOp*
_output_shapes
: *E

debug_name75RMSprop/velocity/separable_conv2d_1/depthwise_kernel/*
dtype0*
shape: *E
shared_name64RMSprop/velocity/separable_conv2d_1/depthwise_kernel
�
HRMSprop/velocity/separable_conv2d_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp4RMSprop/velocity/separable_conv2d_1/depthwise_kernel*&
_output_shapes
: *
dtype0
�
+RMSprop/velocity/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *<

debug_name.,RMSprop/velocity/batch_normalization_1/beta/*
dtype0*
shape: *<
shared_name-+RMSprop/velocity/batch_normalization_1/beta
�
?RMSprop/velocity/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp+RMSprop/velocity/batch_normalization_1/beta*
_output_shapes
: *
dtype0
�
,RMSprop/velocity/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *=

debug_name/-RMSprop/velocity/batch_normalization_1/gamma/*
dtype0*
shape: *=
shared_name.,RMSprop/velocity/batch_normalization_1/gamma
�
@RMSprop/velocity/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp,RMSprop/velocity/batch_normalization_1/gamma*
_output_shapes
: *
dtype0
�
!RMSprop/velocity/conv2d_10/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"RMSprop/velocity/conv2d_10/kernel/*
dtype0*
shape: *2
shared_name#!RMSprop/velocity/conv2d_10/kernel
�
5RMSprop/velocity/conv2d_10/kernel/Read/ReadVariableOpReadVariableOp!RMSprop/velocity/conv2d_10/kernel*&
_output_shapes
: *
dtype0
�
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
dense_2/biasVarHandleOp*
_output_shapes
: *

debug_namedense_2/bias/*
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
�
dense_2/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_2/kernel/*
dtype0*
shape:	�*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	�*
dtype0
�
conv2d_15/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv2d_15/kernel/*
dtype0*
shape:��*!
shared_nameconv2d_15/kernel

$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*(
_output_shapes
:��*
dtype0
�
#separable_conv2d_8/pointwise_kernelVarHandleOp*
_output_shapes
: *4

debug_name&$separable_conv2d_8/pointwise_kernel/*
dtype0*
shape:��*4
shared_name%#separable_conv2d_8/pointwise_kernel
�
7separable_conv2d_8/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_8/pointwise_kernel*(
_output_shapes
:��*
dtype0
�
#separable_conv2d_8/depthwise_kernelVarHandleOp*
_output_shapes
: *4

debug_name&$separable_conv2d_8/depthwise_kernel/*
dtype0*
shape:�*4
shared_name%#separable_conv2d_8/depthwise_kernel
�
7separable_conv2d_8/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_8/depthwise_kernel*'
_output_shapes
:�*
dtype0
�
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *6

debug_name(&batch_normalization_8/moving_variance/*
dtype0*
shape:�*6
shared_name'%batch_normalization_8/moving_variance
�
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *2

debug_name$"batch_normalization_8/moving_mean/*
dtype0*
shape:�*2
shared_name#!batch_normalization_8/moving_mean
�
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *+

debug_namebatch_normalization_8/beta/*
dtype0*
shape:�*+
shared_namebatch_normalization_8/beta
�
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_8/gamma/*
dtype0*
shape:�*,
shared_namebatch_normalization_8/gamma
�
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes	
:�*
dtype0
�
#separable_conv2d_9/pointwise_kernelVarHandleOp*
_output_shapes
: *4

debug_name&$separable_conv2d_9/pointwise_kernel/*
dtype0*
shape:��*4
shared_name%#separable_conv2d_9/pointwise_kernel
�
7separable_conv2d_9/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_9/pointwise_kernel*(
_output_shapes
:��*
dtype0
�
#separable_conv2d_9/depthwise_kernelVarHandleOp*
_output_shapes
: *4

debug_name&$separable_conv2d_9/depthwise_kernel/*
dtype0*
shape:�*4
shared_name%#separable_conv2d_9/depthwise_kernel
�
7separable_conv2d_9/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_9/depthwise_kernel*'
_output_shapes
:�*
dtype0
�
%batch_normalization_9/moving_varianceVarHandleOp*
_output_shapes
: *6

debug_name(&batch_normalization_9/moving_variance/*
dtype0*
shape:�*6
shared_name'%batch_normalization_9/moving_variance
�
9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_9/moving_meanVarHandleOp*
_output_shapes
: *2

debug_name$"batch_normalization_9/moving_mean/*
dtype0*
shape:�*2
shared_name#!batch_normalization_9/moving_mean
�
5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_9/betaVarHandleOp*
_output_shapes
: *+

debug_namebatch_normalization_9/beta/*
dtype0*
shape:�*+
shared_namebatch_normalization_9/beta
�
.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_9/gammaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_9/gamma/*
dtype0*
shape:�*,
shared_namebatch_normalization_9/gamma
�
/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
_output_shapes	
:�*
dtype0
�
conv2d_14/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv2d_14/kernel/*
dtype0*
shape:��*!
shared_nameconv2d_14/kernel

$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*(
_output_shapes
:��*
dtype0
�
#separable_conv2d_6/pointwise_kernelVarHandleOp*
_output_shapes
: *4

debug_name&$separable_conv2d_6/pointwise_kernel/*
dtype0*
shape:��*4
shared_name%#separable_conv2d_6/pointwise_kernel
�
7separable_conv2d_6/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_6/pointwise_kernel*(
_output_shapes
:��*
dtype0
�
#separable_conv2d_6/depthwise_kernelVarHandleOp*
_output_shapes
: *4

debug_name&$separable_conv2d_6/depthwise_kernel/*
dtype0*
shape:�*4
shared_name%#separable_conv2d_6/depthwise_kernel
�
7separable_conv2d_6/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_6/depthwise_kernel*'
_output_shapes
:�*
dtype0
�
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *6

debug_name(&batch_normalization_6/moving_variance/*
dtype0*
shape:�*6
shared_name'%batch_normalization_6/moving_variance
�
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *2

debug_name$"batch_normalization_6/moving_mean/*
dtype0*
shape:�*2
shared_name#!batch_normalization_6/moving_mean
�
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *+

debug_namebatch_normalization_6/beta/*
dtype0*
shape:�*+
shared_namebatch_normalization_6/beta
�
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_6/gamma/*
dtype0*
shape:�*,
shared_namebatch_normalization_6/gamma
�
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes	
:�*
dtype0
�
#separable_conv2d_7/pointwise_kernelVarHandleOp*
_output_shapes
: *4

debug_name&$separable_conv2d_7/pointwise_kernel/*
dtype0*
shape:��*4
shared_name%#separable_conv2d_7/pointwise_kernel
�
7separable_conv2d_7/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_7/pointwise_kernel*(
_output_shapes
:��*
dtype0
�
#separable_conv2d_7/depthwise_kernelVarHandleOp*
_output_shapes
: *4

debug_name&$separable_conv2d_7/depthwise_kernel/*
dtype0*
shape:�*4
shared_name%#separable_conv2d_7/depthwise_kernel
�
7separable_conv2d_7/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_7/depthwise_kernel*'
_output_shapes
:�*
dtype0
�
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *6

debug_name(&batch_normalization_7/moving_variance/*
dtype0*
shape:�*6
shared_name'%batch_normalization_7/moving_variance
�
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *2

debug_name$"batch_normalization_7/moving_mean/*
dtype0*
shape:�*2
shared_name#!batch_normalization_7/moving_mean
�
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *+

debug_namebatch_normalization_7/beta/*
dtype0*
shape:�*+
shared_namebatch_normalization_7/beta
�
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_7/gamma/*
dtype0*
shape:�*,
shared_namebatch_normalization_7/gamma
�
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes	
:�*
dtype0
�
conv2d_13/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv2d_13/kernel/*
dtype0*
shape:@�*!
shared_nameconv2d_13/kernel
~
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*'
_output_shapes
:@�*
dtype0
�
#separable_conv2d_4/pointwise_kernelVarHandleOp*
_output_shapes
: *4

debug_name&$separable_conv2d_4/pointwise_kernel/*
dtype0*
shape:��*4
shared_name%#separable_conv2d_4/pointwise_kernel
�
7separable_conv2d_4/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_4/pointwise_kernel*(
_output_shapes
:��*
dtype0
�
#separable_conv2d_4/depthwise_kernelVarHandleOp*
_output_shapes
: *4

debug_name&$separable_conv2d_4/depthwise_kernel/*
dtype0*
shape:�*4
shared_name%#separable_conv2d_4/depthwise_kernel
�
7separable_conv2d_4/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_4/depthwise_kernel*'
_output_shapes
:�*
dtype0
�
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *6

debug_name(&batch_normalization_4/moving_variance/*
dtype0*
shape:�*6
shared_name'%batch_normalization_4/moving_variance
�
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes	
:�*
dtype0
�
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *2

debug_name$"batch_normalization_4/moving_mean/*
dtype0*
shape:�*2
shared_name#!batch_normalization_4/moving_mean
�
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes	
:�*
dtype0
�
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *+

debug_namebatch_normalization_4/beta/*
dtype0*
shape:�*+
shared_namebatch_normalization_4/beta
�
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_4/gamma/*
dtype0*
shape:�*,
shared_namebatch_normalization_4/gamma
�
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes	
:�*
dtype0
�
#separable_conv2d_5/pointwise_kernelVarHandleOp*
_output_shapes
: *4

debug_name&$separable_conv2d_5/pointwise_kernel/*
dtype0*
shape:@�*4
shared_name%#separable_conv2d_5/pointwise_kernel
�
7separable_conv2d_5/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_5/pointwise_kernel*'
_output_shapes
:@�*
dtype0
�
#separable_conv2d_5/depthwise_kernelVarHandleOp*
_output_shapes
: *4

debug_name&$separable_conv2d_5/depthwise_kernel/*
dtype0*
shape:@*4
shared_name%#separable_conv2d_5/depthwise_kernel
�
7separable_conv2d_5/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_5/depthwise_kernel*&
_output_shapes
:@*
dtype0
�
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *6

debug_name(&batch_normalization_5/moving_variance/*
dtype0*
shape:@*6
shared_name'%batch_normalization_5/moving_variance
�
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0
�
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *2

debug_name$"batch_normalization_5/moving_mean/*
dtype0*
shape:@*2
shared_name#!batch_normalization_5/moving_mean
�
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
�
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *+

debug_namebatch_normalization_5/beta/*
dtype0*
shape:@*+
shared_namebatch_normalization_5/beta
�
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:@*
dtype0
�
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_5/gamma/*
dtype0*
shape:@*,
shared_namebatch_normalization_5/gamma
�
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:@*
dtype0
�
conv2d_12/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv2d_12/kernel/*
dtype0*
shape: @*!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
: @*
dtype0
�
#separable_conv2d_2/pointwise_kernelVarHandleOp*
_output_shapes
: *4

debug_name&$separable_conv2d_2/pointwise_kernel/*
dtype0*
shape:@@*4
shared_name%#separable_conv2d_2/pointwise_kernel
�
7separable_conv2d_2/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_2/pointwise_kernel*&
_output_shapes
:@@*
dtype0
�
#separable_conv2d_2/depthwise_kernelVarHandleOp*
_output_shapes
: *4

debug_name&$separable_conv2d_2/depthwise_kernel/*
dtype0*
shape:@*4
shared_name%#separable_conv2d_2/depthwise_kernel
�
7separable_conv2d_2/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_2/depthwise_kernel*&
_output_shapes
:@*
dtype0
�
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *6

debug_name(&batch_normalization_2/moving_variance/*
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
�
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *2

debug_name$"batch_normalization_2/moving_mean/*
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
�
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *+

debug_namebatch_normalization_2/beta/*
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
�
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_2/gamma/*
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
�
#separable_conv2d_3/pointwise_kernelVarHandleOp*
_output_shapes
: *4

debug_name&$separable_conv2d_3/pointwise_kernel/*
dtype0*
shape: @*4
shared_name%#separable_conv2d_3/pointwise_kernel
�
7separable_conv2d_3/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_3/pointwise_kernel*&
_output_shapes
: @*
dtype0
�
#separable_conv2d_3/depthwise_kernelVarHandleOp*
_output_shapes
: *4

debug_name&$separable_conv2d_3/depthwise_kernel/*
dtype0*
shape: *4
shared_name%#separable_conv2d_3/depthwise_kernel
�
7separable_conv2d_3/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_3/depthwise_kernel*&
_output_shapes
: *
dtype0
�
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *6

debug_name(&batch_normalization_3/moving_variance/*
dtype0*
shape: *6
shared_name'%batch_normalization_3/moving_variance
�
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0
�
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *2

debug_name$"batch_normalization_3/moving_mean/*
dtype0*
shape: *2
shared_name#!batch_normalization_3/moving_mean
�
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0
�
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *+

debug_namebatch_normalization_3/beta/*
dtype0*
shape: *+
shared_namebatch_normalization_3/beta
�
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
: *
dtype0
�
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_3/gamma/*
dtype0*
shape: *,
shared_namebatch_normalization_3/gamma
�
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
: *
dtype0
�
conv2d_11/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv2d_11/kernel/*
dtype0*
shape:  *!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
:  *
dtype0
�
!separable_conv2d/pointwise_kernelVarHandleOp*
_output_shapes
: *2

debug_name$"separable_conv2d/pointwise_kernel/*
dtype0*
shape:  *2
shared_name#!separable_conv2d/pointwise_kernel
�
5separable_conv2d/pointwise_kernel/Read/ReadVariableOpReadVariableOp!separable_conv2d/pointwise_kernel*&
_output_shapes
:  *
dtype0
�
!separable_conv2d/depthwise_kernelVarHandleOp*
_output_shapes
: *2

debug_name$"separable_conv2d/depthwise_kernel/*
dtype0*
shape: *2
shared_name#!separable_conv2d/depthwise_kernel
�
5separable_conv2d/depthwise_kernel/Read/ReadVariableOpReadVariableOp!separable_conv2d/depthwise_kernel*&
_output_shapes
: *
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *4

debug_name&$batch_normalization/moving_variance/*
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *0

debug_name" batch_normalization/moving_mean/*
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *)

debug_namebatch_normalization/beta/*
dtype0*
shape: *)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: **

debug_namebatch_normalization/gamma/*
dtype0*
shape: **
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0
�
#separable_conv2d_1/pointwise_kernelVarHandleOp*
_output_shapes
: *4

debug_name&$separable_conv2d_1/pointwise_kernel/*
dtype0*
shape:  *4
shared_name%#separable_conv2d_1/pointwise_kernel
�
7separable_conv2d_1/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_1/pointwise_kernel*&
_output_shapes
:  *
dtype0
�
#separable_conv2d_1/depthwise_kernelVarHandleOp*
_output_shapes
: *4

debug_name&$separable_conv2d_1/depthwise_kernel/*
dtype0*
shape: *4
shared_name%#separable_conv2d_1/depthwise_kernel
�
7separable_conv2d_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_1/depthwise_kernel*&
_output_shapes
: *
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *6

debug_name(&batch_normalization_1/moving_variance/*
dtype0*
shape: *6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *2

debug_name$"batch_normalization_1/moving_mean/*
dtype0*
shape: *2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *+

debug_namebatch_normalization_1/beta/*
dtype0*
shape: *+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
: *
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_1/gamma/*
dtype0*
shape: *,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
: *
dtype0
�
conv2d_10/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv2d_10/kernel/*
dtype0*
shape: *!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
: *
dtype0
�
serving_default_input_3Placeholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3conv2d_10/kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variance!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelconv2d_11/kernelbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance#separable_conv2d_3/depthwise_kernel#separable_conv2d_3/pointwise_kernelbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance#separable_conv2d_2/depthwise_kernel#separable_conv2d_2/pointwise_kernelconv2d_12/kernelbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance#separable_conv2d_5/depthwise_kernel#separable_conv2d_5/pointwise_kernelbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance#separable_conv2d_4/depthwise_kernel#separable_conv2d_4/pointwise_kernelconv2d_13/kernelbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variance#separable_conv2d_7/depthwise_kernel#separable_conv2d_7/pointwise_kernelbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variance#separable_conv2d_6/depthwise_kernel#separable_conv2d_6/pointwise_kernelconv2d_14/kernelbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_variance#separable_conv2d_9/depthwise_kernel#separable_conv2d_9/pointwise_kernelbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variance#separable_conv2d_8/depthwise_kernel#separable_conv2d_8/pointwise_kernelconv2d_15/kerneldense_2/kerneldense_2/bias*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_183384

NoOpNoOp
ē
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value�B� B�
�
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer-17
layer_with_weights-9
layer-18
layer-19
layer_with_weights-10
layer-20
layer-21
layer_with_weights-11
layer-22
layer-23
layer_with_weights-12
layer-24
layer_with_weights-13
layer-25
layer-26
layer_with_weights-14
layer-27
layer-28
layer_with_weights-15
layer-29
layer-30
 layer_with_weights-16
 layer-31
!layer-32
"layer_with_weights-17
"layer-33
#layer_with_weights-18
#layer-34
$layer-35
%layer_with_weights-19
%layer-36
&layer-37
'layer_with_weights-20
'layer-38
(layer-39
)layer_with_weights-21
)layer-40
*layer-41
+layer_with_weights-22
+layer-42
,layer_with_weights-23
,layer-43
-layer-44
.layer_with_weights-24
.layer-45
/layer-46
0layer_with_weights-25
0layer-47
1layer-48
2layer-49
3layer-50
4layer_with_weights-26
4layer-51
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;_default_save_signature
<	optimizer
=
signatures*
* 
�
>layer-0
?layer-1
@layer-2
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses*
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses* 
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
 T_jit_compiled_convolution_op*
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[axis
	\gamma
]beta
^moving_mean
_moving_variance*
�
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses* 
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses
ldepthwise_kernel
mpointwise_kernel
 n_jit_compiled_convolution_op*
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses
uaxis
	vgamma
wbeta
xmoving_mean
ymoving_variance*
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
�pointwise_kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
�pointwise_kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
�pointwise_kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
�pointwise_kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
�pointwise_kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
�pointwise_kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
�pointwise_kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
�pointwise_kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
�pointwise_kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
S0
\1
]2
^3
_4
l5
m6
v7
w8
x9
y10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67*
�
S0
\1
]2
l3
m4
v5
w6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
;_default_save_signature
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�_velocities
�
_momentums
�_average_gradients
�_update_step_xla*

�serving_default* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

S0*

S0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
\0
]1
^2
_3*

\0
]1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

l0
m1*

l0
m1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
}w
VARIABLE_VALUE#separable_conv2d_1/depthwise_kernel@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE#separable_conv2d_1/pointwise_kernel@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
v0
w1
x2
y3*

v0
w1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
{u
VARIABLE_VALUE!separable_conv2d/depthwise_kernel@layer_with_weights-4/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE!separable_conv2d/pointwise_kernel@layer_with_weights-4/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
}w
VARIABLE_VALUE#separable_conv2d_3/depthwise_kernel@layer_with_weights-7/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE#separable_conv2d_3/pointwise_kernel@layer_with_weights-7/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
}w
VARIABLE_VALUE#separable_conv2d_2/depthwise_kernel@layer_with_weights-9/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE#separable_conv2d_2/pointwise_kernel@layer_with_weights-9/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_12/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
~x
VARIABLE_VALUE#separable_conv2d_5/depthwise_kernelAlayer_with_weights-12/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE#separable_conv2d_5/pointwise_kernelAlayer_with_weights-12/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_4/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_4/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_4/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_4/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
~x
VARIABLE_VALUE#separable_conv2d_4/depthwise_kernelAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE#separable_conv2d_4/pointwise_kernelAlayer_with_weights-14/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_13/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_7/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_7/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_7/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_7/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
~x
VARIABLE_VALUE#separable_conv2d_7/depthwise_kernelAlayer_with_weights-17/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE#separable_conv2d_7/pointwise_kernelAlayer_with_weights-17/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_6/gamma6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_6/beta5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_6/moving_mean<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_6/moving_variance@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
~x
VARIABLE_VALUE#separable_conv2d_6/depthwise_kernelAlayer_with_weights-19/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE#separable_conv2d_6/pointwise_kernelAlayer_with_weights-19/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_14/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_9/gamma6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_9/beta5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_9/moving_mean<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_9/moving_variance@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
~x
VARIABLE_VALUE#separable_conv2d_9/depthwise_kernelAlayer_with_weights-22/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE#separable_conv2d_9/pointwise_kernelAlayer_with_weights-22/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_8/gamma6layer_with_weights-23/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_8/beta5layer_with_weights-23/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE!batch_normalization_8/moving_mean<layer_with_weights-23/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE%batch_normalization_8/moving_variance@layer_with_weights-23/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
~x
VARIABLE_VALUE#separable_conv2d_8/depthwise_kernelAlayer_with_weights-24/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE#separable_conv2d_8/pointwise_kernelAlayer_with_weights-24/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEconv2d_15/kernel7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_2/kernel7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_2/bias5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUE*
�
^0
_1
x2
y3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19*
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451*

�0
�1*
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 

�
_generator*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 

�
_generator*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 

�
_generator*
* 

>0
?1
@2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

^0
_1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

x0
y1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
lf
VARIABLE_VALUE!RMSprop/velocity/conv2d_10/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE,RMSprop/velocity/batch_normalization_1/gamma1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE+RMSprop/velocity/batch_normalization_1/beta1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE4RMSprop/velocity/separable_conv2d_1/depthwise_kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE4RMSprop/velocity/separable_conv2d_1/pointwise_kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*RMSprop/velocity/batch_normalization/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)RMSprop/velocity/batch_normalization/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE2RMSprop/velocity/separable_conv2d/depthwise_kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE2RMSprop/velocity/separable_conv2d/pointwise_kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!RMSprop/velocity/conv2d_11/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE,RMSprop/velocity/batch_normalization_3/gamma2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+RMSprop/velocity/batch_normalization_3/beta2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4RMSprop/velocity/separable_conv2d_3/depthwise_kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4RMSprop/velocity/separable_conv2d_3/pointwise_kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE,RMSprop/velocity/batch_normalization_2/gamma2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+RMSprop/velocity/batch_normalization_2/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4RMSprop/velocity/separable_conv2d_2/depthwise_kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4RMSprop/velocity/separable_conv2d_2/pointwise_kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!RMSprop/velocity/conv2d_12/kernel2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE,RMSprop/velocity/batch_normalization_5/gamma2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+RMSprop/velocity/batch_normalization_5/beta2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4RMSprop/velocity/separable_conv2d_5/depthwise_kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4RMSprop/velocity/separable_conv2d_5/pointwise_kernel2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE,RMSprop/velocity/batch_normalization_4/gamma2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+RMSprop/velocity/batch_normalization_4/beta2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4RMSprop/velocity/separable_conv2d_4/depthwise_kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4RMSprop/velocity/separable_conv2d_4/pointwise_kernel2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!RMSprop/velocity/conv2d_13/kernel2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE,RMSprop/velocity/batch_normalization_7/gamma2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+RMSprop/velocity/batch_normalization_7/beta2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4RMSprop/velocity/separable_conv2d_7/depthwise_kernel2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4RMSprop/velocity/separable_conv2d_7/pointwise_kernel2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE,RMSprop/velocity/batch_normalization_6/gamma2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+RMSprop/velocity/batch_normalization_6/beta2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4RMSprop/velocity/separable_conv2d_6/depthwise_kernel2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4RMSprop/velocity/separable_conv2d_6/pointwise_kernel2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!RMSprop/velocity/conv2d_14/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE,RMSprop/velocity/batch_normalization_9/gamma2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+RMSprop/velocity/batch_normalization_9/beta2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4RMSprop/velocity/separable_conv2d_9/depthwise_kernel2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4RMSprop/velocity/separable_conv2d_9/pointwise_kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE,RMSprop/velocity/batch_normalization_8/gamma2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE+RMSprop/velocity/batch_normalization_8/beta2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4RMSprop/velocity/separable_conv2d_8/depthwise_kernel2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE4RMSprop/velocity/separable_conv2d_8/pointwise_kernel2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!RMSprop/velocity/conv2d_15/kernel2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUERMSprop/velocity/dense_2/kernel2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUERMSprop/velocity/dense_2/bias2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 

�
_state_var*
* 
* 
* 
* 
* 
* 
* 
* 
* 

�
_state_var*
* 
* 
* 
* 
* 
* 
* 
* 
* 

�
_state_var*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
�|
VARIABLE_VALUErandom_flip_2/StateVarRlayer-1/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUErandom_rotation_2/StateVarRlayer-1/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUErandom_zoom_2/StateVarRlayer-1/layer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�&
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_10/kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variance!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelconv2d_11/kernelbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance#separable_conv2d_3/depthwise_kernel#separable_conv2d_3/pointwise_kernelbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance#separable_conv2d_2/depthwise_kernel#separable_conv2d_2/pointwise_kernelconv2d_12/kernelbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance#separable_conv2d_5/depthwise_kernel#separable_conv2d_5/pointwise_kernelbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance#separable_conv2d_4/depthwise_kernel#separable_conv2d_4/pointwise_kernelconv2d_13/kernelbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variance#separable_conv2d_7/depthwise_kernel#separable_conv2d_7/pointwise_kernelbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variance#separable_conv2d_6/depthwise_kernel#separable_conv2d_6/pointwise_kernelconv2d_14/kernelbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_variance#separable_conv2d_9/depthwise_kernel#separable_conv2d_9/pointwise_kernelbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variance#separable_conv2d_8/depthwise_kernel#separable_conv2d_8/pointwise_kernelconv2d_15/kerneldense_2/kerneldense_2/bias	iterationlearning_rate!RMSprop/velocity/conv2d_10/kernel,RMSprop/velocity/batch_normalization_1/gamma+RMSprop/velocity/batch_normalization_1/beta4RMSprop/velocity/separable_conv2d_1/depthwise_kernel4RMSprop/velocity/separable_conv2d_1/pointwise_kernel*RMSprop/velocity/batch_normalization/gamma)RMSprop/velocity/batch_normalization/beta2RMSprop/velocity/separable_conv2d/depthwise_kernel2RMSprop/velocity/separable_conv2d/pointwise_kernel!RMSprop/velocity/conv2d_11/kernel,RMSprop/velocity/batch_normalization_3/gamma+RMSprop/velocity/batch_normalization_3/beta4RMSprop/velocity/separable_conv2d_3/depthwise_kernel4RMSprop/velocity/separable_conv2d_3/pointwise_kernel,RMSprop/velocity/batch_normalization_2/gamma+RMSprop/velocity/batch_normalization_2/beta4RMSprop/velocity/separable_conv2d_2/depthwise_kernel4RMSprop/velocity/separable_conv2d_2/pointwise_kernel!RMSprop/velocity/conv2d_12/kernel,RMSprop/velocity/batch_normalization_5/gamma+RMSprop/velocity/batch_normalization_5/beta4RMSprop/velocity/separable_conv2d_5/depthwise_kernel4RMSprop/velocity/separable_conv2d_5/pointwise_kernel,RMSprop/velocity/batch_normalization_4/gamma+RMSprop/velocity/batch_normalization_4/beta4RMSprop/velocity/separable_conv2d_4/depthwise_kernel4RMSprop/velocity/separable_conv2d_4/pointwise_kernel!RMSprop/velocity/conv2d_13/kernel,RMSprop/velocity/batch_normalization_7/gamma+RMSprop/velocity/batch_normalization_7/beta4RMSprop/velocity/separable_conv2d_7/depthwise_kernel4RMSprop/velocity/separable_conv2d_7/pointwise_kernel,RMSprop/velocity/batch_normalization_6/gamma+RMSprop/velocity/batch_normalization_6/beta4RMSprop/velocity/separable_conv2d_6/depthwise_kernel4RMSprop/velocity/separable_conv2d_6/pointwise_kernel!RMSprop/velocity/conv2d_14/kernel,RMSprop/velocity/batch_normalization_9/gamma+RMSprop/velocity/batch_normalization_9/beta4RMSprop/velocity/separable_conv2d_9/depthwise_kernel4RMSprop/velocity/separable_conv2d_9/pointwise_kernel,RMSprop/velocity/batch_normalization_8/gamma+RMSprop/velocity/batch_normalization_8/beta4RMSprop/velocity/separable_conv2d_8/depthwise_kernel4RMSprop/velocity/separable_conv2d_8/pointwise_kernel!RMSprop/velocity/conv2d_15/kernelRMSprop/velocity/dense_2/kernelRMSprop/velocity/dense_2/biastotal_1count_1totalcountrandom_flip_2/StateVarrandom_rotation_2/StateVarrandom_zoom_2/StateVarConst*�
Tin�
�2*
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
GPU 2J 8� *(
f#R!
__inference__traced_save_185692
�&
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_10/kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variance!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelconv2d_11/kernelbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variance#separable_conv2d_3/depthwise_kernel#separable_conv2d_3/pointwise_kernelbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance#separable_conv2d_2/depthwise_kernel#separable_conv2d_2/pointwise_kernelconv2d_12/kernelbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance#separable_conv2d_5/depthwise_kernel#separable_conv2d_5/pointwise_kernelbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance#separable_conv2d_4/depthwise_kernel#separable_conv2d_4/pointwise_kernelconv2d_13/kernelbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variance#separable_conv2d_7/depthwise_kernel#separable_conv2d_7/pointwise_kernelbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variance#separable_conv2d_6/depthwise_kernel#separable_conv2d_6/pointwise_kernelconv2d_14/kernelbatch_normalization_9/gammabatch_normalization_9/beta!batch_normalization_9/moving_mean%batch_normalization_9/moving_variance#separable_conv2d_9/depthwise_kernel#separable_conv2d_9/pointwise_kernelbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variance#separable_conv2d_8/depthwise_kernel#separable_conv2d_8/pointwise_kernelconv2d_15/kerneldense_2/kerneldense_2/bias	iterationlearning_rate!RMSprop/velocity/conv2d_10/kernel,RMSprop/velocity/batch_normalization_1/gamma+RMSprop/velocity/batch_normalization_1/beta4RMSprop/velocity/separable_conv2d_1/depthwise_kernel4RMSprop/velocity/separable_conv2d_1/pointwise_kernel*RMSprop/velocity/batch_normalization/gamma)RMSprop/velocity/batch_normalization/beta2RMSprop/velocity/separable_conv2d/depthwise_kernel2RMSprop/velocity/separable_conv2d/pointwise_kernel!RMSprop/velocity/conv2d_11/kernel,RMSprop/velocity/batch_normalization_3/gamma+RMSprop/velocity/batch_normalization_3/beta4RMSprop/velocity/separable_conv2d_3/depthwise_kernel4RMSprop/velocity/separable_conv2d_3/pointwise_kernel,RMSprop/velocity/batch_normalization_2/gamma+RMSprop/velocity/batch_normalization_2/beta4RMSprop/velocity/separable_conv2d_2/depthwise_kernel4RMSprop/velocity/separable_conv2d_2/pointwise_kernel!RMSprop/velocity/conv2d_12/kernel,RMSprop/velocity/batch_normalization_5/gamma+RMSprop/velocity/batch_normalization_5/beta4RMSprop/velocity/separable_conv2d_5/depthwise_kernel4RMSprop/velocity/separable_conv2d_5/pointwise_kernel,RMSprop/velocity/batch_normalization_4/gamma+RMSprop/velocity/batch_normalization_4/beta4RMSprop/velocity/separable_conv2d_4/depthwise_kernel4RMSprop/velocity/separable_conv2d_4/pointwise_kernel!RMSprop/velocity/conv2d_13/kernel,RMSprop/velocity/batch_normalization_7/gamma+RMSprop/velocity/batch_normalization_7/beta4RMSprop/velocity/separable_conv2d_7/depthwise_kernel4RMSprop/velocity/separable_conv2d_7/pointwise_kernel,RMSprop/velocity/batch_normalization_6/gamma+RMSprop/velocity/batch_normalization_6/beta4RMSprop/velocity/separable_conv2d_6/depthwise_kernel4RMSprop/velocity/separable_conv2d_6/pointwise_kernel!RMSprop/velocity/conv2d_14/kernel,RMSprop/velocity/batch_normalization_9/gamma+RMSprop/velocity/batch_normalization_9/beta4RMSprop/velocity/separable_conv2d_9/depthwise_kernel4RMSprop/velocity/separable_conv2d_9/pointwise_kernel,RMSprop/velocity/batch_normalization_8/gamma+RMSprop/velocity/batch_normalization_8/beta4RMSprop/velocity/separable_conv2d_8/depthwise_kernel4RMSprop/velocity/separable_conv2d_8/pointwise_kernel!RMSprop/velocity/conv2d_15/kernelRMSprop/velocity/dense_2/kernelRMSprop/velocity/dense_2/biastotal_1count_1totalcountrandom_flip_2/StateVarrandom_rotation_2/StateVarrandom_zoom_2/StateVar*�
Tin�
�2~*
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
GPU 2J 8� *+
f&R$
"__inference__traced_restore_186076�*
�	
�
3__inference_separable_conv2d_8_layer_call_fn_184473

inputs"
unknown:�%
	unknown_0:��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_8_layer_call_and_return_conditional_losses_182230�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184469:&"
 
_user_specified_name184467:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
R
&__inference_add_2_layer_call_fn_184071
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_182461i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������  �:���������  �:ZV
0
_output_shapes
:���������  �
"
_user_specified_name
inputs_1:Z V
0
_output_shapes
:���������  �
"
_user_specified_name
inputs_0
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_184010

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_12_layer_call_and_return_conditional_losses_183843

inputs8
conv2d_readvariableop_resource: @
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������??@*
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:���������??@:
NoOpNoOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������~~ : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������~~ 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_183770

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
N
2__inference_random_rotation_2_layer_call_fn_184665

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_random_rotation_2_layer_call_and_return_conditional_losses_181309j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�"
C__inference_model_2_layer_call_and_return_conditional_losses_182806
input_3*
conv2d_10_182615: *
batch_normalization_1_182618: *
batch_normalization_1_182620: *
batch_normalization_1_182622: *
batch_normalization_1_182624: 3
separable_conv2d_1_182628: 3
separable_conv2d_1_182630:  (
batch_normalization_182633: (
batch_normalization_182635: (
batch_normalization_182637: (
batch_normalization_182639: 1
separable_conv2d_182643: 1
separable_conv2d_182645:  *
conv2d_11_182649:  *
batch_normalization_3_182653: *
batch_normalization_3_182655: *
batch_normalization_3_182657: *
batch_normalization_3_182659: 3
separable_conv2d_3_182663: 3
separable_conv2d_3_182665: @*
batch_normalization_2_182668:@*
batch_normalization_2_182670:@*
batch_normalization_2_182672:@*
batch_normalization_2_182674:@3
separable_conv2d_2_182678:@3
separable_conv2d_2_182680:@@*
conv2d_12_182684: @*
batch_normalization_5_182688:@*
batch_normalization_5_182690:@*
batch_normalization_5_182692:@*
batch_normalization_5_182694:@3
separable_conv2d_5_182698:@4
separable_conv2d_5_182700:@�+
batch_normalization_4_182703:	�+
batch_normalization_4_182705:	�+
batch_normalization_4_182707:	�+
batch_normalization_4_182709:	�4
separable_conv2d_4_182713:�5
separable_conv2d_4_182715:��+
conv2d_13_182719:@�+
batch_normalization_7_182723:	�+
batch_normalization_7_182725:	�+
batch_normalization_7_182727:	�+
batch_normalization_7_182729:	�4
separable_conv2d_7_182733:�5
separable_conv2d_7_182735:��+
batch_normalization_6_182738:	�+
batch_normalization_6_182740:	�+
batch_normalization_6_182742:	�+
batch_normalization_6_182744:	�4
separable_conv2d_6_182748:�5
separable_conv2d_6_182750:��,
conv2d_14_182754:��+
batch_normalization_9_182758:	�+
batch_normalization_9_182760:	�+
batch_normalization_9_182762:	�+
batch_normalization_9_182764:	�4
separable_conv2d_9_182768:�5
separable_conv2d_9_182770:��+
batch_normalization_8_182773:	�+
batch_normalization_8_182775:	�+
batch_normalization_8_182777:	�+
batch_normalization_8_182779:	�4
separable_conv2d_8_182783:�5
separable_conv2d_8_182785:��,
conv2d_15_182789:��!
dense_2_182800:	�
dense_2_182802:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�-batch_normalization_8/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall�!conv2d_10/StatefulPartitionedCall�!conv2d_11/StatefulPartitionedCall�!conv2d_12/StatefulPartitionedCall�!conv2d_13/StatefulPartitionedCall�!conv2d_14/StatefulPartitionedCall�!conv2d_15/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�(separable_conv2d/StatefulPartitionedCall�*separable_conv2d_1/StatefulPartitionedCall�*separable_conv2d_2/StatefulPartitionedCall�*separable_conv2d_3/StatefulPartitionedCall�*separable_conv2d_4/StatefulPartitionedCall�*separable_conv2d_5/StatefulPartitionedCall�*separable_conv2d_6/StatefulPartitionedCall�*separable_conv2d_7/StatefulPartitionedCall�*separable_conv2d_8/StatefulPartitionedCall�*separable_conv2d_9/StatefulPartitionedCall�
sequential_2/PartitionedCallPartitionedCallinput_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_181317�
rescaling_2/PartitionedCallPartitionedCall%sequential_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_rescaling_2_layer_call_and_return_conditional_losses_182277�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall$rescaling_2/PartitionedCall:output:0conv2d_10_182615*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_182285�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_1_182618batch_normalization_1_182620batch_normalization_1_182622batch_normalization_1_182624*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_181405�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_182302�
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0separable_conv2d_1_182628separable_conv2d_1_182630*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_181443�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0batch_normalization_182633batch_normalization_182635batch_normalization_182637batch_normalization_182639*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_181488�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_182322�
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0separable_conv2d_182643separable_conv2d_182645*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_separable_conv2d_layer_call_and_return_conditional_losses_181526�
max_pooling2d_8/PartitionedCallPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_181540�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_182649*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_182336�
add/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_182345�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0batch_normalization_3_182653batch_normalization_3_182655batch_normalization_3_182657batch_normalization_3_182659*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~ *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_181581�
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_182360�
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0separable_conv2d_3_182663separable_conv2d_3_182665*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_181619�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0batch_normalization_2_182668batch_normalization_2_182670batch_normalization_2_182672batch_normalization_2_182674*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_181664�
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_182380�
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0separable_conv2d_2_182678separable_conv2d_2_182680*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_181702�
max_pooling2d_9/PartitionedCallPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������??@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_181716�
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0conv2d_12_182684*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������??@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_182394�
add_1/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������??@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_182403�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0batch_normalization_5_182688batch_normalization_5_182690batch_normalization_5_182692batch_normalization_5_182694*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������??@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_181757�
activation_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������??@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_182418�
*separable_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0separable_conv2d_5_182698separable_conv2d_5_182700*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������??�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_181795�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_5/StatefulPartitionedCall:output:0batch_normalization_4_182703batch_normalization_4_182705batch_normalization_4_182707batch_normalization_4_182709*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������??�*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_181840�
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������??�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_182438�
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0separable_conv2d_4_182713separable_conv2d_4_182715*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������??�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_181878�
 max_pooling2d_10/PartitionedCallPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_181892�
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0conv2d_13_182719*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_182452�
add_2/PartitionedCallPartitionedCall)max_pooling2d_10/PartitionedCall:output:0*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_182461�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0batch_normalization_7_182723batch_normalization_7_182725batch_normalization_7_182727batch_normalization_7_182729*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_181933�
activation_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_7_layer_call_and_return_conditional_losses_182476�
*separable_conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0separable_conv2d_7_182733separable_conv2d_7_182735*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_181971�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_7/StatefulPartitionedCall:output:0batch_normalization_6_182738batch_normalization_6_182740batch_normalization_6_182742batch_normalization_6_182744*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_182016�
activation_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_182496�
*separable_conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0separable_conv2d_6_182748separable_conv2d_6_182750*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_182054�
 max_pooling2d_11/PartitionedCallPartitionedCall3separable_conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_182068�
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0conv2d_14_182754*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_182510�
add_3/PartitionedCallPartitionedCall)max_pooling2d_11/PartitionedCall:output:0*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_182519�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0batch_normalization_9_182758batch_normalization_9_182760batch_normalization_9_182762batch_normalization_9_182764*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_182109�
activation_9/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_9_layer_call_and_return_conditional_losses_182534�
*separable_conv2d_9/StatefulPartitionedCallStatefulPartitionedCall%activation_9/PartitionedCall:output:0separable_conv2d_9_182768separable_conv2d_9_182770*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_9_layer_call_and_return_conditional_losses_182147�
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_9/StatefulPartitionedCall:output:0batch_normalization_8_182773batch_normalization_8_182775batch_normalization_8_182777batch_normalization_8_182779*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_182192�
activation_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_8_layer_call_and_return_conditional_losses_182554�
*separable_conv2d_8/StatefulPartitionedCallStatefulPartitionedCall%activation_8/PartitionedCall:output:0separable_conv2d_8_182783separable_conv2d_8_182785*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_8_layer_call_and_return_conditional_losses_182230�
 max_pooling2d_12/PartitionedCallPartitionedCall3separable_conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_182244�
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0conv2d_15_182789*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_182568�
add_4/PartitionedCallPartitionedCall)max_pooling2d_12/PartitionedCall:output:0*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_4_layer_call_and_return_conditional_losses_182577�
(global_average_pooling2d/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_182255�
dropout_1/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_182798�
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_182800dense_2_182802*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_182603w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall+^separable_conv2d_5/StatefulPartitionedCall+^separable_conv2d_6/StatefulPartitionedCall+^separable_conv2d_7/StatefulPartitionedCall+^separable_conv2d_8/StatefulPartitionedCall+^separable_conv2d_9/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall2X
*separable_conv2d_5/StatefulPartitionedCall*separable_conv2d_5/StatefulPartitionedCall2X
*separable_conv2d_6/StatefulPartitionedCall*separable_conv2d_6/StatefulPartitionedCall2X
*separable_conv2d_7/StatefulPartitionedCall*separable_conv2d_7/StatefulPartitionedCall2X
*separable_conv2d_8/StatefulPartitionedCall*separable_conv2d_8/StatefulPartitionedCall2X
*separable_conv2d_9/StatefulPartitionedCall*separable_conv2d_9/StatefulPartitionedCall:&D"
 
_user_specified_name182802:&C"
 
_user_specified_name182800:&B"
 
_user_specified_name182789:&A"
 
_user_specified_name182785:&@"
 
_user_specified_name182783:&?"
 
_user_specified_name182779:&>"
 
_user_specified_name182777:&="
 
_user_specified_name182775:&<"
 
_user_specified_name182773:&;"
 
_user_specified_name182770:&:"
 
_user_specified_name182768:&9"
 
_user_specified_name182764:&8"
 
_user_specified_name182762:&7"
 
_user_specified_name182760:&6"
 
_user_specified_name182758:&5"
 
_user_specified_name182754:&4"
 
_user_specified_name182750:&3"
 
_user_specified_name182748:&2"
 
_user_specified_name182744:&1"
 
_user_specified_name182742:&0"
 
_user_specified_name182740:&/"
 
_user_specified_name182738:&."
 
_user_specified_name182735:&-"
 
_user_specified_name182733:&,"
 
_user_specified_name182729:&+"
 
_user_specified_name182727:&*"
 
_user_specified_name182725:&)"
 
_user_specified_name182723:&("
 
_user_specified_name182719:&'"
 
_user_specified_name182715:&&"
 
_user_specified_name182713:&%"
 
_user_specified_name182709:&$"
 
_user_specified_name182707:&#"
 
_user_specified_name182705:&""
 
_user_specified_name182703:&!"
 
_user_specified_name182700:& "
 
_user_specified_name182698:&"
 
_user_specified_name182694:&"
 
_user_specified_name182692:&"
 
_user_specified_name182690:&"
 
_user_specified_name182688:&"
 
_user_specified_name182684:&"
 
_user_specified_name182680:&"
 
_user_specified_name182678:&"
 
_user_specified_name182674:&"
 
_user_specified_name182672:&"
 
_user_specified_name182670:&"
 
_user_specified_name182668:&"
 
_user_specified_name182665:&"
 
_user_specified_name182663:&"
 
_user_specified_name182659:&"
 
_user_specified_name182657:&"
 
_user_specified_name182655:&"
 
_user_specified_name182653:&"
 
_user_specified_name182649:&"
 
_user_specified_name182645:&"
 
_user_specified_name182643:&"
 
_user_specified_name182639:&
"
 
_user_specified_name182637:&	"
 
_user_specified_name182635:&"
 
_user_specified_name182633:&"
 
_user_specified_name182630:&"
 
_user_specified_name182628:&"
 
_user_specified_name182624:&"
 
_user_specified_name182622:&"
 
_user_specified_name182620:&"
 
_user_specified_name182618:&"
 
_user_specified_name182615:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_3
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_182603

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_181757

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
E__inference_conv2d_12_layer_call_and_return_conditional_losses_182394

inputs8
conv2d_readvariableop_resource: @
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������??@*
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:���������??@:
NoOpNoOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������~~ : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������~~ 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_183677

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
I
-__inference_activation_7_layer_call_fn_184144

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_7_layer_call_and_return_conditional_losses_182476i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������  �:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�	
�
3__inference_separable_conv2d_7_layer_call_fn_184158

inputs"
unknown:�%
	unknown_0:��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_181971�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184154:&"
 
_user_specified_name184152:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
I
-__inference_activation_2_layer_call_fn_183793

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_182380h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������~~@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������~~@:W S
/
_output_shapes
:���������~~@
 
_user_specified_nameinputs
�
�
N__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_184263

inputsC
(separable_conv2d_readvariableop_resource:�F
*separable_conv2d_readvariableop_1_resource:��
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:��*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
e
I__inference_random_flip_2_layer_call_and_return_conditional_losses_181304

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_182591

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
m
A__inference_add_3_layer_call_and_return_conditional_losses_184299
inputs_0
inputs_1
identity[
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:����������X
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������:����������:ZV
0
_output_shapes
:����������
"
_user_specified_name
inputs_1:Z V
0
_output_shapes
:����������
"
_user_specified_name
inputs_0
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_181470

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
d
H__inference_activation_4_layer_call_and_return_conditional_losses_182438

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:���������??�c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:���������??�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������??�:X T
0
_output_shapes
:���������??�
 
_user_specified_nameinputs
�
d
H__inference_activation_9_layer_call_and_return_conditional_losses_184371

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:����������c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_181878

inputsC
(separable_conv2d_readvariableop_resource:�F
*separable_conv2d_readvariableop_1_resource:��
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:��*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      �      o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_181540

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_182244

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_183788

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
(__inference_dense_2_layer_call_fn_184568

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_182603o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184564:&"
 
_user_specified_name184562:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_183548

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
Ȁ
�
M__inference_random_rotation_2_layer_call_and_return_conditional_losses_184783

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity��stateful_uniform/RngReadAndSkipI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|� �Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|� ?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:�
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:���������z
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: �
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:����������
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:���������Z
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?g
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ^
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:���������\
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: |
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:���������^
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:���������\
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ~
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:���������~
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:���������~
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:���������^
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:���������\
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: `
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:���������\
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: �
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:���������`
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:���������\
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: �
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:����������
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:���������~
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:���������`
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:���������g
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
::��m
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:���������v
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:���������v
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_maskv
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:���������v
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:���������v
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:���������v
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_maskv
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:���������]
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������S
transform/ShapeShapeinputs*
T0*
_output_shapes
::��g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:�����������*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR�
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:�����������D
NoOpNoOp ^stateful_uniform/RngReadAndSkip*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:($
"
_user_specified_name
resource:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
N__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_181795

inputsB
(separable_conv2d_readvariableop_resource:@E
*separable_conv2d_readvariableop_1_resource:@�
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@�*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_181716

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
N__inference_separable_conv2d_8_layer_call_and_return_conditional_losses_182230

inputsC
(separable_conv2d_readvariableop_resource:�F
*separable_conv2d_readvariableop_1_resource:��
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:��*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_181739

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
c
G__inference_rescaling_2_layer_call_and_return_conditional_losses_182277

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:�����������d
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:�����������Y
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
I
-__inference_activation_6_layer_call_fn_184237

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_182496i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������  �:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
V
-__inference_sequential_2_layer_call_fn_181333
random_flip_2_input
identity�
PartitionedCallPartitionedCallrandom_flip_2_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_181317j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:f b
1
_output_shapes
:�����������
-
_user_specified_namerandom_flip_2_input
�
�
N__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_183504

inputsB
(separable_conv2d_readvariableop_resource: D
*separable_conv2d_readvariableop_1_resource:  
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:  *
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
G
+__inference_activation_layer_call_fn_183571

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_182322j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:����������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:����������� :Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�
�
E__inference_conv2d_13_layer_call_and_return_conditional_losses_182452

inputs9
conv2d_readvariableop_resource:@�
identity��Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:���������  �:
NoOpNoOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������??@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������??@
 
_user_specified_nameinputs
�
�
1__inference_separable_conv2d_layer_call_fn_183585

inputs!
unknown: #
	unknown_0:  
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_separable_conv2d_layer_call_and_return_conditional_losses_181526�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183581:&"
 
_user_specified_name183579:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
d
H__inference_activation_5_layer_call_and_return_conditional_losses_183927

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������??@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������??@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������??@:W S
/
_output_shapes
:���������??@
 
_user_specified_nameinputs
�
I
-__inference_activation_5_layer_call_fn_183922

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������??@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_182418h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������??@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������??@:W S
/
_output_shapes
:���������??@
 
_user_specified_nameinputs
�
�
E__inference_conv2d_15_layer_call_and_return_conditional_losses_184509

inputs:
conv2d_readvariableop_resource:��
identity��Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:����������:
NoOpNoOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
H__inference_activation_1_layer_call_and_return_conditional_losses_182302

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:����������� d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:����������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:����������� :Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�
P
$__inference_add_layer_call_fn_183627
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_182345h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������~~ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������~~ :���������~~ :YU
/
_output_shapes
:���������~~ 
"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������~~ 
"
_user_specified_name
inputs_0
�
g
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_183607

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
R
&__inference_add_3_layer_call_fn_184293
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_182519i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������:����������:ZV
0
_output_shapes
:����������
"
_user_specified_name
inputs_1:Z V
0
_output_shapes
:����������
"
_user_specified_name
inputs_0
�
d
H__inference_activation_7_layer_call_and_return_conditional_losses_184149

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:���������  �c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������  �:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
�
3__inference_separable_conv2d_2_layer_call_fn_183807

inputs!
unknown:@#
	unknown_0:@@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_181702�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183803:&"
 
_user_specified_name183801:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
I
-__inference_activation_8_layer_call_fn_184459

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_8_layer_call_and_return_conditional_losses_182554i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_2_layer_call_fn_183752

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_181664�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183748:&"
 
_user_specified_name183746:&"
 
_user_specified_name183744:&"
 
_user_specified_name183742:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_183899

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_182109

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_15_layer_call_and_return_conditional_losses_182568

inputs:
conv2d_readvariableop_resource:��
identity��Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:����������:
NoOpNoOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_181840

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_184139

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
͹
�J
!__inference__wrapped_model_181007
input_3J
0model_2_conv2d_10_conv2d_readvariableop_resource: C
5model_2_batch_normalization_1_readvariableop_resource: E
7model_2_batch_normalization_1_readvariableop_1_resource: T
Fmodel_2_batch_normalization_1_fusedbatchnormv3_readvariableop_resource: V
Hmodel_2_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: ]
Cmodel_2_separable_conv2d_1_separable_conv2d_readvariableop_resource: _
Emodel_2_separable_conv2d_1_separable_conv2d_readvariableop_1_resource:  A
3model_2_batch_normalization_readvariableop_resource: C
5model_2_batch_normalization_readvariableop_1_resource: R
Dmodel_2_batch_normalization_fusedbatchnormv3_readvariableop_resource: T
Fmodel_2_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: [
Amodel_2_separable_conv2d_separable_conv2d_readvariableop_resource: ]
Cmodel_2_separable_conv2d_separable_conv2d_readvariableop_1_resource:  J
0model_2_conv2d_11_conv2d_readvariableop_resource:  C
5model_2_batch_normalization_3_readvariableop_resource: E
7model_2_batch_normalization_3_readvariableop_1_resource: T
Fmodel_2_batch_normalization_3_fusedbatchnormv3_readvariableop_resource: V
Hmodel_2_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource: ]
Cmodel_2_separable_conv2d_3_separable_conv2d_readvariableop_resource: _
Emodel_2_separable_conv2d_3_separable_conv2d_readvariableop_1_resource: @C
5model_2_batch_normalization_2_readvariableop_resource:@E
7model_2_batch_normalization_2_readvariableop_1_resource:@T
Fmodel_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@V
Hmodel_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@]
Cmodel_2_separable_conv2d_2_separable_conv2d_readvariableop_resource:@_
Emodel_2_separable_conv2d_2_separable_conv2d_readvariableop_1_resource:@@J
0model_2_conv2d_12_conv2d_readvariableop_resource: @C
5model_2_batch_normalization_5_readvariableop_resource:@E
7model_2_batch_normalization_5_readvariableop_1_resource:@T
Fmodel_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:@V
Hmodel_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:@]
Cmodel_2_separable_conv2d_5_separable_conv2d_readvariableop_resource:@`
Emodel_2_separable_conv2d_5_separable_conv2d_readvariableop_1_resource:@�D
5model_2_batch_normalization_4_readvariableop_resource:	�F
7model_2_batch_normalization_4_readvariableop_1_resource:	�U
Fmodel_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	�W
Hmodel_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	�^
Cmodel_2_separable_conv2d_4_separable_conv2d_readvariableop_resource:�a
Emodel_2_separable_conv2d_4_separable_conv2d_readvariableop_1_resource:��K
0model_2_conv2d_13_conv2d_readvariableop_resource:@�D
5model_2_batch_normalization_7_readvariableop_resource:	�F
7model_2_batch_normalization_7_readvariableop_1_resource:	�U
Fmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource:	�W
Hmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource:	�^
Cmodel_2_separable_conv2d_7_separable_conv2d_readvariableop_resource:�a
Emodel_2_separable_conv2d_7_separable_conv2d_readvariableop_1_resource:��D
5model_2_batch_normalization_6_readvariableop_resource:	�F
7model_2_batch_normalization_6_readvariableop_1_resource:	�U
Fmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:	�W
Hmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:	�^
Cmodel_2_separable_conv2d_6_separable_conv2d_readvariableop_resource:�a
Emodel_2_separable_conv2d_6_separable_conv2d_readvariableop_1_resource:��L
0model_2_conv2d_14_conv2d_readvariableop_resource:��D
5model_2_batch_normalization_9_readvariableop_resource:	�F
7model_2_batch_normalization_9_readvariableop_1_resource:	�U
Fmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource:	�W
Hmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource:	�^
Cmodel_2_separable_conv2d_9_separable_conv2d_readvariableop_resource:�a
Emodel_2_separable_conv2d_9_separable_conv2d_readvariableop_1_resource:��D
5model_2_batch_normalization_8_readvariableop_resource:	�F
7model_2_batch_normalization_8_readvariableop_1_resource:	�U
Fmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_resource:	�W
Hmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource:	�^
Cmodel_2_separable_conv2d_8_separable_conv2d_readvariableop_resource:�a
Emodel_2_separable_conv2d_8_separable_conv2d_readvariableop_1_resource:��L
0model_2_conv2d_15_conv2d_readvariableop_resource:��A
.model_2_dense_2_matmul_readvariableop_resource:	�=
/model_2_dense_2_biasadd_readvariableop_resource:
identity��;model_2/batch_normalization/FusedBatchNormV3/ReadVariableOp�=model_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�*model_2/batch_normalization/ReadVariableOp�,model_2/batch_normalization/ReadVariableOp_1�=model_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�?model_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�,model_2/batch_normalization_1/ReadVariableOp�.model_2/batch_normalization_1/ReadVariableOp_1�=model_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�?model_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�,model_2/batch_normalization_2/ReadVariableOp�.model_2/batch_normalization_2/ReadVariableOp_1�=model_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�?model_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�,model_2/batch_normalization_3/ReadVariableOp�.model_2/batch_normalization_3/ReadVariableOp_1�=model_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp�?model_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�,model_2/batch_normalization_4/ReadVariableOp�.model_2/batch_normalization_4/ReadVariableOp_1�=model_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp�?model_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�,model_2/batch_normalization_5/ReadVariableOp�.model_2/batch_normalization_5/ReadVariableOp_1�=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�?model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�,model_2/batch_normalization_6/ReadVariableOp�.model_2/batch_normalization_6/ReadVariableOp_1�=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp�?model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�,model_2/batch_normalization_7/ReadVariableOp�.model_2/batch_normalization_7/ReadVariableOp_1�=model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp�?model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1�,model_2/batch_normalization_8/ReadVariableOp�.model_2/batch_normalization_8/ReadVariableOp_1�=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp�?model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1�,model_2/batch_normalization_9/ReadVariableOp�.model_2/batch_normalization_9/ReadVariableOp_1�'model_2/conv2d_10/Conv2D/ReadVariableOp�'model_2/conv2d_11/Conv2D/ReadVariableOp�'model_2/conv2d_12/Conv2D/ReadVariableOp�'model_2/conv2d_13/Conv2D/ReadVariableOp�'model_2/conv2d_14/Conv2D/ReadVariableOp�'model_2/conv2d_15/Conv2D/ReadVariableOp�&model_2/dense_2/BiasAdd/ReadVariableOp�%model_2/dense_2/MatMul/ReadVariableOp�8model_2/separable_conv2d/separable_conv2d/ReadVariableOp�:model_2/separable_conv2d/separable_conv2d/ReadVariableOp_1�:model_2/separable_conv2d_1/separable_conv2d/ReadVariableOp�<model_2/separable_conv2d_1/separable_conv2d/ReadVariableOp_1�:model_2/separable_conv2d_2/separable_conv2d/ReadVariableOp�<model_2/separable_conv2d_2/separable_conv2d/ReadVariableOp_1�:model_2/separable_conv2d_3/separable_conv2d/ReadVariableOp�<model_2/separable_conv2d_3/separable_conv2d/ReadVariableOp_1�:model_2/separable_conv2d_4/separable_conv2d/ReadVariableOp�<model_2/separable_conv2d_4/separable_conv2d/ReadVariableOp_1�:model_2/separable_conv2d_5/separable_conv2d/ReadVariableOp�<model_2/separable_conv2d_5/separable_conv2d/ReadVariableOp_1�:model_2/separable_conv2d_6/separable_conv2d/ReadVariableOp�<model_2/separable_conv2d_6/separable_conv2d/ReadVariableOp_1�:model_2/separable_conv2d_7/separable_conv2d/ReadVariableOp�<model_2/separable_conv2d_7/separable_conv2d/ReadVariableOp_1�:model_2/separable_conv2d_8/separable_conv2d/ReadVariableOp�<model_2/separable_conv2d_8/separable_conv2d/ReadVariableOp_1�:model_2/separable_conv2d_9/separable_conv2d/ReadVariableOp�<model_2/separable_conv2d_9/separable_conv2d/ReadVariableOp_1_
model_2/rescaling_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;a
model_2/rescaling_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    �
model_2/rescaling_2/mulMulinput_3#model_2/rescaling_2/Cast/x:output:0*
T0*1
_output_shapes
:������������
model_2/rescaling_2/addAddV2model_2/rescaling_2/mul:z:0%model_2/rescaling_2/Cast_1/x:output:0*
T0*1
_output_shapes
:������������
'model_2/conv2d_10/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
model_2/conv2d_10/Conv2DConv2Dmodel_2/rescaling_2/add:z:0/model_2/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingVALID*
strides
�
,model_2/batch_normalization_1/ReadVariableOpReadVariableOp5model_2_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype0�
.model_2/batch_normalization_1/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype0�
=model_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
?model_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
.model_2/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3!model_2/conv2d_10/Conv2D:output:04model_2/batch_normalization_1/ReadVariableOp:value:06model_2/batch_normalization_1/ReadVariableOp_1:value:0Emodel_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:����������� : : : : :*
epsilon%o�:*
is_training( �
model_2/activation_1/ReluRelu2model_2/batch_normalization_1/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:����������� �
:model_2/separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOpCmodel_2_separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
<model_2/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOpEmodel_2_separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:  *
dtype0�
1model_2/separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
9model_2/separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
5model_2/separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative'model_2/activation_1/Relu:activations:0Bmodel_2/separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
+model_2/separable_conv2d_1/separable_conv2dConv2D>model_2/separable_conv2d_1/separable_conv2d/depthwise:output:0Dmodel_2/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:����������� *
paddingVALID*
strides
�
*model_2/batch_normalization/ReadVariableOpReadVariableOp3model_2_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0�
,model_2/batch_normalization/ReadVariableOp_1ReadVariableOp5model_2_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0�
;model_2/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_2_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
=model_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_2_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
,model_2/batch_normalization/FusedBatchNormV3FusedBatchNormV34model_2/separable_conv2d_1/separable_conv2d:output:02model_2/batch_normalization/ReadVariableOp:value:04model_2/batch_normalization/ReadVariableOp_1:value:0Cmodel_2/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Emodel_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:����������� : : : : :*
epsilon%o�:*
is_training( �
model_2/activation/ReluRelu0model_2/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:����������� �
8model_2/separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOpAmodel_2_separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
:model_2/separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOpCmodel_2_separable_conv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:  *
dtype0�
/model_2/separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
7model_2/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
3model_2/separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNative%model_2/activation/Relu:activations:0@model_2/separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
)model_2/separable_conv2d/separable_conv2dConv2D<model_2/separable_conv2d/separable_conv2d/depthwise:output:0Bmodel_2/separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:����������� *
paddingVALID*
strides
�
model_2/max_pooling2d_8/MaxPoolMaxPool2model_2/separable_conv2d/separable_conv2d:output:0*/
_output_shapes
:���������~~ *
ksize
*
paddingSAME*
strides
�
'model_2/conv2d_11/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
model_2/conv2d_11/Conv2DConv2D!model_2/conv2d_10/Conv2D:output:0/model_2/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������~~ *
paddingSAME*
strides
�
model_2/add/addAddV2(model_2/max_pooling2d_8/MaxPool:output:0!model_2/conv2d_11/Conv2D:output:0*
T0*/
_output_shapes
:���������~~ �
,model_2/batch_normalization_3/ReadVariableOpReadVariableOp5model_2_batch_normalization_3_readvariableop_resource*
_output_shapes
: *
dtype0�
.model_2/batch_normalization_3/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
=model_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
?model_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
.model_2/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3model_2/add/add:z:04model_2/batch_normalization_3/ReadVariableOp:value:06model_2/batch_normalization_3/ReadVariableOp_1:value:0Emodel_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������~~ : : : : :*
epsilon%o�:*
is_training( �
model_2/activation_3/ReluRelu2model_2/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������~~ �
:model_2/separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOpCmodel_2_separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
<model_2/separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOpEmodel_2_separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
: @*
dtype0�
1model_2/separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
9model_2/separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
5model_2/separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNative'model_2/activation_3/Relu:activations:0Bmodel_2/separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������~~ *
paddingSAME*
strides
�
+model_2/separable_conv2d_3/separable_conv2dConv2D>model_2/separable_conv2d_3/separable_conv2d/depthwise:output:0Dmodel_2/separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:���������~~@*
paddingVALID*
strides
�
,model_2/batch_normalization_2/ReadVariableOpReadVariableOp5model_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
.model_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
=model_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
?model_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
.model_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV34model_2/separable_conv2d_3/separable_conv2d:output:04model_2/batch_normalization_2/ReadVariableOp:value:06model_2/batch_normalization_2/ReadVariableOp_1:value:0Emodel_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������~~@:@:@:@:@:*
epsilon%o�:*
is_training( �
model_2/activation_2/ReluRelu2model_2/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������~~@�
:model_2/separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOpCmodel_2_separable_conv2d_2_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
<model_2/separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOpEmodel_2_separable_conv2d_2_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:@@*
dtype0�
1model_2/separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      �
9model_2/separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
5model_2/separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNative'model_2/activation_2/Relu:activations:0Bmodel_2/separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������~~@*
paddingSAME*
strides
�
+model_2/separable_conv2d_2/separable_conv2dConv2D>model_2/separable_conv2d_2/separable_conv2d/depthwise:output:0Dmodel_2/separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*/
_output_shapes
:���������~~@*
paddingVALID*
strides
�
model_2/max_pooling2d_9/MaxPoolMaxPool4model_2/separable_conv2d_2/separable_conv2d:output:0*/
_output_shapes
:���������??@*
ksize
*
paddingSAME*
strides
�
'model_2/conv2d_12/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
model_2/conv2d_12/Conv2DConv2Dmodel_2/add/add:z:0/model_2/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������??@*
paddingSAME*
strides
�
model_2/add_1/addAddV2(model_2/max_pooling2d_9/MaxPool:output:0!model_2/conv2d_12/Conv2D:output:0*
T0*/
_output_shapes
:���������??@�
,model_2/batch_normalization_5/ReadVariableOpReadVariableOp5model_2_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype0�
.model_2/batch_normalization_5/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
=model_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
?model_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
.model_2/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3model_2/add_1/add:z:04model_2/batch_normalization_5/ReadVariableOp:value:06model_2/batch_normalization_5/ReadVariableOp_1:value:0Emodel_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������??@:@:@:@:@:*
epsilon%o�:*
is_training( �
model_2/activation_5/ReluRelu2model_2/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������??@�
:model_2/separable_conv2d_5/separable_conv2d/ReadVariableOpReadVariableOpCmodel_2_separable_conv2d_5_separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
<model_2/separable_conv2d_5/separable_conv2d/ReadVariableOp_1ReadVariableOpEmodel_2_separable_conv2d_5_separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@�*
dtype0�
1model_2/separable_conv2d_5/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      �
9model_2/separable_conv2d_5/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
5model_2/separable_conv2d_5/separable_conv2d/depthwiseDepthwiseConv2dNative'model_2/activation_5/Relu:activations:0Bmodel_2/separable_conv2d_5/separable_conv2d/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������??@*
paddingSAME*
strides
�
+model_2/separable_conv2d_5/separable_conv2dConv2D>model_2/separable_conv2d_5/separable_conv2d/depthwise:output:0Dmodel_2/separable_conv2d_5/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:���������??�*
paddingVALID*
strides
�
,model_2/batch_normalization_4/ReadVariableOpReadVariableOp5model_2_batch_normalization_4_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.model_2/batch_normalization_4/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
=model_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
?model_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
.model_2/batch_normalization_4/FusedBatchNormV3FusedBatchNormV34model_2/separable_conv2d_5/separable_conv2d:output:04model_2/batch_normalization_4/ReadVariableOp:value:06model_2/batch_normalization_4/ReadVariableOp_1:value:0Emodel_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������??�:�:�:�:�:*
epsilon%o�:*
is_training( �
model_2/activation_4/ReluRelu2model_2/batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:���������??��
:model_2/separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOpCmodel_2_separable_conv2d_4_separable_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
<model_2/separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOpEmodel_2_separable_conv2d_4_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:��*
dtype0�
1model_2/separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      �      �
9model_2/separable_conv2d_4/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
5model_2/separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNative'model_2/activation_4/Relu:activations:0Bmodel_2/separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������??�*
paddingSAME*
strides
�
+model_2/separable_conv2d_4/separable_conv2dConv2D>model_2/separable_conv2d_4/separable_conv2d/depthwise:output:0Dmodel_2/separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:���������??�*
paddingVALID*
strides
�
 model_2/max_pooling2d_10/MaxPoolMaxPool4model_2/separable_conv2d_4/separable_conv2d:output:0*0
_output_shapes
:���������  �*
ksize
*
paddingSAME*
strides
�
'model_2/conv2d_13/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_13_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
model_2/conv2d_13/Conv2DConv2Dmodel_2/add_1/add:z:0/model_2/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
model_2/add_2/addAddV2)model_2/max_pooling2d_10/MaxPool:output:0!model_2/conv2d_13/Conv2D:output:0*
T0*0
_output_shapes
:���������  ��
,model_2/batch_normalization_7/ReadVariableOpReadVariableOp5model_2_batch_normalization_7_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.model_2/batch_normalization_7/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_7_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
?model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
.model_2/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3model_2/add_2/add:z:04model_2/batch_normalization_7/ReadVariableOp:value:06model_2/batch_normalization_7/ReadVariableOp_1:value:0Emodel_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������  �:�:�:�:�:*
epsilon%o�:*
is_training( �
model_2/activation_7/ReluRelu2model_2/batch_normalization_7/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:���������  ��
:model_2/separable_conv2d_7/separable_conv2d/ReadVariableOpReadVariableOpCmodel_2_separable_conv2d_7_separable_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
<model_2/separable_conv2d_7/separable_conv2d/ReadVariableOp_1ReadVariableOpEmodel_2_separable_conv2d_7_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:��*
dtype0�
1model_2/separable_conv2d_7/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      �      �
9model_2/separable_conv2d_7/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
5model_2/separable_conv2d_7/separable_conv2d/depthwiseDepthwiseConv2dNative'model_2/activation_7/Relu:activations:0Bmodel_2/separable_conv2d_7/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
+model_2/separable_conv2d_7/separable_conv2dConv2D>model_2/separable_conv2d_7/separable_conv2d/depthwise:output:0Dmodel_2/separable_conv2d_7/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:���������  �*
paddingVALID*
strides
�
,model_2/batch_normalization_6/ReadVariableOpReadVariableOp5model_2_batch_normalization_6_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.model_2/batch_normalization_6/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_6_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
?model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
.model_2/batch_normalization_6/FusedBatchNormV3FusedBatchNormV34model_2/separable_conv2d_7/separable_conv2d:output:04model_2/batch_normalization_6/ReadVariableOp:value:06model_2/batch_normalization_6/ReadVariableOp_1:value:0Emodel_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������  �:�:�:�:�:*
epsilon%o�:*
is_training( �
model_2/activation_6/ReluRelu2model_2/batch_normalization_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:���������  ��
:model_2/separable_conv2d_6/separable_conv2d/ReadVariableOpReadVariableOpCmodel_2_separable_conv2d_6_separable_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
<model_2/separable_conv2d_6/separable_conv2d/ReadVariableOp_1ReadVariableOpEmodel_2_separable_conv2d_6_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:��*
dtype0�
1model_2/separable_conv2d_6/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
9model_2/separable_conv2d_6/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
5model_2/separable_conv2d_6/separable_conv2d/depthwiseDepthwiseConv2dNative'model_2/activation_6/Relu:activations:0Bmodel_2/separable_conv2d_6/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
�
+model_2/separable_conv2d_6/separable_conv2dConv2D>model_2/separable_conv2d_6/separable_conv2d/depthwise:output:0Dmodel_2/separable_conv2d_6/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:���������  �*
paddingVALID*
strides
�
 model_2/max_pooling2d_11/MaxPoolMaxPool4model_2/separable_conv2d_6/separable_conv2d:output:0*0
_output_shapes
:����������*
ksize
*
paddingSAME*
strides
�
'model_2/conv2d_14/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_14_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model_2/conv2d_14/Conv2DConv2Dmodel_2/add_2/add:z:0/model_2/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
model_2/add_3/addAddV2)model_2/max_pooling2d_11/MaxPool:output:0!model_2/conv2d_14/Conv2D:output:0*
T0*0
_output_shapes
:�����������
,model_2/batch_normalization_9/ReadVariableOpReadVariableOp5model_2_batch_normalization_9_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.model_2/batch_normalization_9/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_9_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
?model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_9_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
.model_2/batch_normalization_9/FusedBatchNormV3FusedBatchNormV3model_2/add_3/add:z:04model_2/batch_normalization_9/ReadVariableOp:value:06model_2/batch_normalization_9/ReadVariableOp_1:value:0Emodel_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
model_2/activation_9/ReluRelu2model_2/batch_normalization_9/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
:model_2/separable_conv2d_9/separable_conv2d/ReadVariableOpReadVariableOpCmodel_2_separable_conv2d_9_separable_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
<model_2/separable_conv2d_9/separable_conv2d/ReadVariableOp_1ReadVariableOpEmodel_2_separable_conv2d_9_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:��*
dtype0�
1model_2/separable_conv2d_9/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
9model_2/separable_conv2d_9/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
5model_2/separable_conv2d_9/separable_conv2d/depthwiseDepthwiseConv2dNative'model_2/activation_9/Relu:activations:0Bmodel_2/separable_conv2d_9/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
+model_2/separable_conv2d_9/separable_conv2dConv2D>model_2/separable_conv2d_9/separable_conv2d/depthwise:output:0Dmodel_2/separable_conv2d_9/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
,model_2/batch_normalization_8/ReadVariableOpReadVariableOp5model_2_batch_normalization_8_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.model_2/batch_normalization_8/ReadVariableOp_1ReadVariableOp7model_2_batch_normalization_8_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
=model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
?model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_2_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
.model_2/batch_normalization_8/FusedBatchNormV3FusedBatchNormV34model_2/separable_conv2d_9/separable_conv2d:output:04model_2/batch_normalization_8/ReadVariableOp:value:06model_2/batch_normalization_8/ReadVariableOp_1:value:0Emodel_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
model_2/activation_8/ReluRelu2model_2/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:�����������
:model_2/separable_conv2d_8/separable_conv2d/ReadVariableOpReadVariableOpCmodel_2_separable_conv2d_8_separable_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
<model_2/separable_conv2d_8/separable_conv2d/ReadVariableOp_1ReadVariableOpEmodel_2_separable_conv2d_8_separable_conv2d_readvariableop_1_resource*(
_output_shapes
:��*
dtype0�
1model_2/separable_conv2d_8/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
9model_2/separable_conv2d_8/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
5model_2/separable_conv2d_8/separable_conv2d/depthwiseDepthwiseConv2dNative'model_2/activation_8/Relu:activations:0Bmodel_2/separable_conv2d_8/separable_conv2d/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
+model_2/separable_conv2d_8/separable_conv2dConv2D>model_2/separable_conv2d_8/separable_conv2d/depthwise:output:0Dmodel_2/separable_conv2d_8/separable_conv2d/ReadVariableOp_1:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
 model_2/max_pooling2d_12/MaxPoolMaxPool4model_2/separable_conv2d_8/separable_conv2d:output:0*0
_output_shapes
:����������*
ksize
*
paddingSAME*
strides
�
'model_2/conv2d_15/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_15_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model_2/conv2d_15/Conv2DConv2Dmodel_2/add_3/add:z:0/model_2/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
model_2/add_4/addAddV2)model_2/max_pooling2d_12/MaxPool:output:0!model_2/conv2d_15/Conv2D:output:0*
T0*0
_output_shapes
:�����������
7model_2/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      �
%model_2/global_average_pooling2d/MeanMeanmodel_2/add_4/add:z:0@model_2/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:�����������
model_2/dropout_1/IdentityIdentity.model_2/global_average_pooling2d/Mean:output:0*
T0*(
_output_shapes
:�����������
%model_2/dense_2/MatMul/ReadVariableOpReadVariableOp.model_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model_2/dense_2/MatMulMatMul#model_2/dropout_1/Identity:output:0-model_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&model_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_2/dense_2/BiasAddBiasAdd model_2/dense_2/MatMul:product:0.model_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
model_2/dense_2/SoftmaxSoftmax model_2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������p
IdentityIdentity!model_2/dense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp<^model_2/batch_normalization/FusedBatchNormV3/ReadVariableOp>^model_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1+^model_2/batch_normalization/ReadVariableOp-^model_2/batch_normalization/ReadVariableOp_1>^model_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1-^model_2/batch_normalization_1/ReadVariableOp/^model_2/batch_normalization_1/ReadVariableOp_1>^model_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1-^model_2/batch_normalization_2/ReadVariableOp/^model_2/batch_normalization_2/ReadVariableOp_1>^model_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1-^model_2/batch_normalization_3/ReadVariableOp/^model_2/batch_normalization_3/ReadVariableOp_1>^model_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1-^model_2/batch_normalization_4/ReadVariableOp/^model_2/batch_normalization_4/ReadVariableOp_1>^model_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1-^model_2/batch_normalization_5/ReadVariableOp/^model_2/batch_normalization_5/ReadVariableOp_1>^model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1-^model_2/batch_normalization_6/ReadVariableOp/^model_2/batch_normalization_6/ReadVariableOp_1>^model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1-^model_2/batch_normalization_7/ReadVariableOp/^model_2/batch_normalization_7/ReadVariableOp_1>^model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1-^model_2/batch_normalization_8/ReadVariableOp/^model_2/batch_normalization_8/ReadVariableOp_1>^model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp@^model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1-^model_2/batch_normalization_9/ReadVariableOp/^model_2/batch_normalization_9/ReadVariableOp_1(^model_2/conv2d_10/Conv2D/ReadVariableOp(^model_2/conv2d_11/Conv2D/ReadVariableOp(^model_2/conv2d_12/Conv2D/ReadVariableOp(^model_2/conv2d_13/Conv2D/ReadVariableOp(^model_2/conv2d_14/Conv2D/ReadVariableOp(^model_2/conv2d_15/Conv2D/ReadVariableOp'^model_2/dense_2/BiasAdd/ReadVariableOp&^model_2/dense_2/MatMul/ReadVariableOp9^model_2/separable_conv2d/separable_conv2d/ReadVariableOp;^model_2/separable_conv2d/separable_conv2d/ReadVariableOp_1;^model_2/separable_conv2d_1/separable_conv2d/ReadVariableOp=^model_2/separable_conv2d_1/separable_conv2d/ReadVariableOp_1;^model_2/separable_conv2d_2/separable_conv2d/ReadVariableOp=^model_2/separable_conv2d_2/separable_conv2d/ReadVariableOp_1;^model_2/separable_conv2d_3/separable_conv2d/ReadVariableOp=^model_2/separable_conv2d_3/separable_conv2d/ReadVariableOp_1;^model_2/separable_conv2d_4/separable_conv2d/ReadVariableOp=^model_2/separable_conv2d_4/separable_conv2d/ReadVariableOp_1;^model_2/separable_conv2d_5/separable_conv2d/ReadVariableOp=^model_2/separable_conv2d_5/separable_conv2d/ReadVariableOp_1;^model_2/separable_conv2d_6/separable_conv2d/ReadVariableOp=^model_2/separable_conv2d_6/separable_conv2d/ReadVariableOp_1;^model_2/separable_conv2d_7/separable_conv2d/ReadVariableOp=^model_2/separable_conv2d_7/separable_conv2d/ReadVariableOp_1;^model_2/separable_conv2d_8/separable_conv2d/ReadVariableOp=^model_2/separable_conv2d_8/separable_conv2d/ReadVariableOp_1;^model_2/separable_conv2d_9/separable_conv2d/ReadVariableOp=^model_2/separable_conv2d_9/separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2~
=model_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_1=model_2/batch_normalization/FusedBatchNormV3/ReadVariableOp_12z
;model_2/batch_normalization/FusedBatchNormV3/ReadVariableOp;model_2/batch_normalization/FusedBatchNormV3/ReadVariableOp2\
,model_2/batch_normalization/ReadVariableOp_1,model_2/batch_normalization/ReadVariableOp_12X
*model_2/batch_normalization/ReadVariableOp*model_2/batch_normalization/ReadVariableOp2�
?model_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12~
=model_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2`
.model_2/batch_normalization_1/ReadVariableOp_1.model_2/batch_normalization_1/ReadVariableOp_12\
,model_2/batch_normalization_1/ReadVariableOp,model_2/batch_normalization_1/ReadVariableOp2�
?model_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12~
=model_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2`
.model_2/batch_normalization_2/ReadVariableOp_1.model_2/batch_normalization_2/ReadVariableOp_12\
,model_2/batch_normalization_2/ReadVariableOp,model_2/batch_normalization_2/ReadVariableOp2�
?model_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12~
=model_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2`
.model_2/batch_normalization_3/ReadVariableOp_1.model_2/batch_normalization_3/ReadVariableOp_12\
,model_2/batch_normalization_3/ReadVariableOp,model_2/batch_normalization_3/ReadVariableOp2�
?model_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12~
=model_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2`
.model_2/batch_normalization_4/ReadVariableOp_1.model_2/batch_normalization_4/ReadVariableOp_12\
,model_2/batch_normalization_4/ReadVariableOp,model_2/batch_normalization_4/ReadVariableOp2�
?model_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12~
=model_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2`
.model_2/batch_normalization_5/ReadVariableOp_1.model_2/batch_normalization_5/ReadVariableOp_12\
,model_2/batch_normalization_5/ReadVariableOp,model_2/batch_normalization_5/ReadVariableOp2�
?model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12~
=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2`
.model_2/batch_normalization_6/ReadVariableOp_1.model_2/batch_normalization_6/ReadVariableOp_12\
,model_2/batch_normalization_6/ReadVariableOp,model_2/batch_normalization_6/ReadVariableOp2�
?model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12~
=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2`
.model_2/batch_normalization_7/ReadVariableOp_1.model_2/batch_normalization_7/ReadVariableOp_12\
,model_2/batch_normalization_7/ReadVariableOp,model_2/batch_normalization_7/ReadVariableOp2�
?model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12~
=model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2`
.model_2/batch_normalization_8/ReadVariableOp_1.model_2/batch_normalization_8/ReadVariableOp_12\
,model_2/batch_normalization_8/ReadVariableOp,model_2/batch_normalization_8/ReadVariableOp2�
?model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_1?model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp_12~
=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp=model_2/batch_normalization_9/FusedBatchNormV3/ReadVariableOp2`
.model_2/batch_normalization_9/ReadVariableOp_1.model_2/batch_normalization_9/ReadVariableOp_12\
,model_2/batch_normalization_9/ReadVariableOp,model_2/batch_normalization_9/ReadVariableOp2R
'model_2/conv2d_10/Conv2D/ReadVariableOp'model_2/conv2d_10/Conv2D/ReadVariableOp2R
'model_2/conv2d_11/Conv2D/ReadVariableOp'model_2/conv2d_11/Conv2D/ReadVariableOp2R
'model_2/conv2d_12/Conv2D/ReadVariableOp'model_2/conv2d_12/Conv2D/ReadVariableOp2R
'model_2/conv2d_13/Conv2D/ReadVariableOp'model_2/conv2d_13/Conv2D/ReadVariableOp2R
'model_2/conv2d_14/Conv2D/ReadVariableOp'model_2/conv2d_14/Conv2D/ReadVariableOp2R
'model_2/conv2d_15/Conv2D/ReadVariableOp'model_2/conv2d_15/Conv2D/ReadVariableOp2P
&model_2/dense_2/BiasAdd/ReadVariableOp&model_2/dense_2/BiasAdd/ReadVariableOp2N
%model_2/dense_2/MatMul/ReadVariableOp%model_2/dense_2/MatMul/ReadVariableOp2x
:model_2/separable_conv2d/separable_conv2d/ReadVariableOp_1:model_2/separable_conv2d/separable_conv2d/ReadVariableOp_12t
8model_2/separable_conv2d/separable_conv2d/ReadVariableOp8model_2/separable_conv2d/separable_conv2d/ReadVariableOp2|
<model_2/separable_conv2d_1/separable_conv2d/ReadVariableOp_1<model_2/separable_conv2d_1/separable_conv2d/ReadVariableOp_12x
:model_2/separable_conv2d_1/separable_conv2d/ReadVariableOp:model_2/separable_conv2d_1/separable_conv2d/ReadVariableOp2|
<model_2/separable_conv2d_2/separable_conv2d/ReadVariableOp_1<model_2/separable_conv2d_2/separable_conv2d/ReadVariableOp_12x
:model_2/separable_conv2d_2/separable_conv2d/ReadVariableOp:model_2/separable_conv2d_2/separable_conv2d/ReadVariableOp2|
<model_2/separable_conv2d_3/separable_conv2d/ReadVariableOp_1<model_2/separable_conv2d_3/separable_conv2d/ReadVariableOp_12x
:model_2/separable_conv2d_3/separable_conv2d/ReadVariableOp:model_2/separable_conv2d_3/separable_conv2d/ReadVariableOp2|
<model_2/separable_conv2d_4/separable_conv2d/ReadVariableOp_1<model_2/separable_conv2d_4/separable_conv2d/ReadVariableOp_12x
:model_2/separable_conv2d_4/separable_conv2d/ReadVariableOp:model_2/separable_conv2d_4/separable_conv2d/ReadVariableOp2|
<model_2/separable_conv2d_5/separable_conv2d/ReadVariableOp_1<model_2/separable_conv2d_5/separable_conv2d/ReadVariableOp_12x
:model_2/separable_conv2d_5/separable_conv2d/ReadVariableOp:model_2/separable_conv2d_5/separable_conv2d/ReadVariableOp2|
<model_2/separable_conv2d_6/separable_conv2d/ReadVariableOp_1<model_2/separable_conv2d_6/separable_conv2d/ReadVariableOp_12x
:model_2/separable_conv2d_6/separable_conv2d/ReadVariableOp:model_2/separable_conv2d_6/separable_conv2d/ReadVariableOp2|
<model_2/separable_conv2d_7/separable_conv2d/ReadVariableOp_1<model_2/separable_conv2d_7/separable_conv2d/ReadVariableOp_12x
:model_2/separable_conv2d_7/separable_conv2d/ReadVariableOp:model_2/separable_conv2d_7/separable_conv2d/ReadVariableOp2|
<model_2/separable_conv2d_8/separable_conv2d/ReadVariableOp_1<model_2/separable_conv2d_8/separable_conv2d/ReadVariableOp_12x
:model_2/separable_conv2d_8/separable_conv2d/ReadVariableOp:model_2/separable_conv2d_8/separable_conv2d/ReadVariableOp2|
<model_2/separable_conv2d_9/separable_conv2d/ReadVariableOp_1<model_2/separable_conv2d_9/separable_conv2d/ReadVariableOp_12x
:model_2/separable_conv2d_9/separable_conv2d/ReadVariableOp:model_2/separable_conv2d_9/separable_conv2d/ReadVariableOp:(D$
"
_user_specified_name
resource:(C$
"
_user_specified_name
resource:(B$
"
_user_specified_name
resource:(A$
"
_user_specified_name
resource:(@$
"
_user_specified_name
resource:(?$
"
_user_specified_name
resource:(>$
"
_user_specified_name
resource:(=$
"
_user_specified_name
resource:(<$
"
_user_specified_name
resource:(;$
"
_user_specified_name
resource:(:$
"
_user_specified_name
resource:(9$
"
_user_specified_name
resource:(8$
"
_user_specified_name
resource:(7$
"
_user_specified_name
resource:(6$
"
_user_specified_name
resource:(5$
"
_user_specified_name
resource:(4$
"
_user_specified_name
resource:(3$
"
_user_specified_name
resource:(2$
"
_user_specified_name
resource:(1$
"
_user_specified_name
resource:(0$
"
_user_specified_name
resource:(/$
"
_user_specified_name
resource:(.$
"
_user_specified_name
resource:(-$
"
_user_specified_name
resource:(,$
"
_user_specified_name
resource:(+$
"
_user_specified_name
resource:(*$
"
_user_specified_name
resource:()$
"
_user_specified_name
resource:(($
"
_user_specified_name
resource:('$
"
_user_specified_name
resource:(&$
"
_user_specified_name
resource:(%$
"
_user_specified_name
resource:($$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_3
�
h
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_184051

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_3_layer_call_fn_183659

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_181581�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183655:&"
 
_user_specified_name183653:&"
 
_user_specified_name183651:&"
 
_user_specified_name183649:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_183992

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�n
�
I__inference_random_zoom_2_layer_call_and_return_conditional_losses_184901

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity��stateful_uniform/RngReadAndSkipI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: Z
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L?Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *���?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:�
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:���������z
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: �
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:����������
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:���������M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2stateful_uniform:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������^
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
::��i
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
zoom_matrix/strided_sliceStridedSlicezoom_matrix/Shape:output:0(zoom_matrix/strided_slice/stack:output:0*zoom_matrix/strided_slice/stack_1:output:0*zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?_
zoom_matrix/subSub
Cast_1:y:0zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: Z
zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:���������x
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:���������X
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
zoom_matrix/sub_2SubCast:y:0zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: \
zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:���������|
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:���������v
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask\
zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
zoom_matrix/zeros/packedPack"zoom_matrix/strided_slice:output:0#zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:���������^
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
zoom_matrix/zeros_1/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������v
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask^
zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
zoom_matrix/zeros_2/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:���������Y
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������S
transform/ShapeShapeinputs*
T0*
_output_shapes
::��g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:�����������*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR�
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:�����������D
NoOpNoOp ^stateful_uniform/RngReadAndSkip*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:($
"
_user_specified_name
resource:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_184361

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_2_layer_call_fn_183739

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_181646�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183735:&"
 
_user_specified_name183733:&"
 
_user_specified_name183731:&"
 
_user_specified_name183729:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
��
�`
"__inference__traced_restore_186076
file_prefix;
!assignvariableop_conv2d_10_kernel: <
.assignvariableop_1_batch_normalization_1_gamma: ;
-assignvariableop_2_batch_normalization_1_beta: B
4assignvariableop_3_batch_normalization_1_moving_mean: F
8assignvariableop_4_batch_normalization_1_moving_variance: P
6assignvariableop_5_separable_conv2d_1_depthwise_kernel: P
6assignvariableop_6_separable_conv2d_1_pointwise_kernel:  :
,assignvariableop_7_batch_normalization_gamma: 9
+assignvariableop_8_batch_normalization_beta: @
2assignvariableop_9_batch_normalization_moving_mean: E
7assignvariableop_10_batch_normalization_moving_variance: O
5assignvariableop_11_separable_conv2d_depthwise_kernel: O
5assignvariableop_12_separable_conv2d_pointwise_kernel:  >
$assignvariableop_13_conv2d_11_kernel:  =
/assignvariableop_14_batch_normalization_3_gamma: <
.assignvariableop_15_batch_normalization_3_beta: C
5assignvariableop_16_batch_normalization_3_moving_mean: G
9assignvariableop_17_batch_normalization_3_moving_variance: Q
7assignvariableop_18_separable_conv2d_3_depthwise_kernel: Q
7assignvariableop_19_separable_conv2d_3_pointwise_kernel: @=
/assignvariableop_20_batch_normalization_2_gamma:@<
.assignvariableop_21_batch_normalization_2_beta:@C
5assignvariableop_22_batch_normalization_2_moving_mean:@G
9assignvariableop_23_batch_normalization_2_moving_variance:@Q
7assignvariableop_24_separable_conv2d_2_depthwise_kernel:@Q
7assignvariableop_25_separable_conv2d_2_pointwise_kernel:@@>
$assignvariableop_26_conv2d_12_kernel: @=
/assignvariableop_27_batch_normalization_5_gamma:@<
.assignvariableop_28_batch_normalization_5_beta:@C
5assignvariableop_29_batch_normalization_5_moving_mean:@G
9assignvariableop_30_batch_normalization_5_moving_variance:@Q
7assignvariableop_31_separable_conv2d_5_depthwise_kernel:@R
7assignvariableop_32_separable_conv2d_5_pointwise_kernel:@�>
/assignvariableop_33_batch_normalization_4_gamma:	�=
.assignvariableop_34_batch_normalization_4_beta:	�D
5assignvariableop_35_batch_normalization_4_moving_mean:	�H
9assignvariableop_36_batch_normalization_4_moving_variance:	�R
7assignvariableop_37_separable_conv2d_4_depthwise_kernel:�S
7assignvariableop_38_separable_conv2d_4_pointwise_kernel:��?
$assignvariableop_39_conv2d_13_kernel:@�>
/assignvariableop_40_batch_normalization_7_gamma:	�=
.assignvariableop_41_batch_normalization_7_beta:	�D
5assignvariableop_42_batch_normalization_7_moving_mean:	�H
9assignvariableop_43_batch_normalization_7_moving_variance:	�R
7assignvariableop_44_separable_conv2d_7_depthwise_kernel:�S
7assignvariableop_45_separable_conv2d_7_pointwise_kernel:��>
/assignvariableop_46_batch_normalization_6_gamma:	�=
.assignvariableop_47_batch_normalization_6_beta:	�D
5assignvariableop_48_batch_normalization_6_moving_mean:	�H
9assignvariableop_49_batch_normalization_6_moving_variance:	�R
7assignvariableop_50_separable_conv2d_6_depthwise_kernel:�S
7assignvariableop_51_separable_conv2d_6_pointwise_kernel:��@
$assignvariableop_52_conv2d_14_kernel:��>
/assignvariableop_53_batch_normalization_9_gamma:	�=
.assignvariableop_54_batch_normalization_9_beta:	�D
5assignvariableop_55_batch_normalization_9_moving_mean:	�H
9assignvariableop_56_batch_normalization_9_moving_variance:	�R
7assignvariableop_57_separable_conv2d_9_depthwise_kernel:�S
7assignvariableop_58_separable_conv2d_9_pointwise_kernel:��>
/assignvariableop_59_batch_normalization_8_gamma:	�=
.assignvariableop_60_batch_normalization_8_beta:	�D
5assignvariableop_61_batch_normalization_8_moving_mean:	�H
9assignvariableop_62_batch_normalization_8_moving_variance:	�R
7assignvariableop_63_separable_conv2d_8_depthwise_kernel:�S
7assignvariableop_64_separable_conv2d_8_pointwise_kernel:��@
$assignvariableop_65_conv2d_15_kernel:��5
"assignvariableop_66_dense_2_kernel:	�.
 assignvariableop_67_dense_2_bias:'
assignvariableop_68_iteration:	 +
!assignvariableop_69_learning_rate: O
5assignvariableop_70_rmsprop_velocity_conv2d_10_kernel: N
@assignvariableop_71_rmsprop_velocity_batch_normalization_1_gamma: M
?assignvariableop_72_rmsprop_velocity_batch_normalization_1_beta: b
Hassignvariableop_73_rmsprop_velocity_separable_conv2d_1_depthwise_kernel: b
Hassignvariableop_74_rmsprop_velocity_separable_conv2d_1_pointwise_kernel:  L
>assignvariableop_75_rmsprop_velocity_batch_normalization_gamma: K
=assignvariableop_76_rmsprop_velocity_batch_normalization_beta: `
Fassignvariableop_77_rmsprop_velocity_separable_conv2d_depthwise_kernel: `
Fassignvariableop_78_rmsprop_velocity_separable_conv2d_pointwise_kernel:  O
5assignvariableop_79_rmsprop_velocity_conv2d_11_kernel:  N
@assignvariableop_80_rmsprop_velocity_batch_normalization_3_gamma: M
?assignvariableop_81_rmsprop_velocity_batch_normalization_3_beta: b
Hassignvariableop_82_rmsprop_velocity_separable_conv2d_3_depthwise_kernel: b
Hassignvariableop_83_rmsprop_velocity_separable_conv2d_3_pointwise_kernel: @N
@assignvariableop_84_rmsprop_velocity_batch_normalization_2_gamma:@M
?assignvariableop_85_rmsprop_velocity_batch_normalization_2_beta:@b
Hassignvariableop_86_rmsprop_velocity_separable_conv2d_2_depthwise_kernel:@b
Hassignvariableop_87_rmsprop_velocity_separable_conv2d_2_pointwise_kernel:@@O
5assignvariableop_88_rmsprop_velocity_conv2d_12_kernel: @N
@assignvariableop_89_rmsprop_velocity_batch_normalization_5_gamma:@M
?assignvariableop_90_rmsprop_velocity_batch_normalization_5_beta:@b
Hassignvariableop_91_rmsprop_velocity_separable_conv2d_5_depthwise_kernel:@c
Hassignvariableop_92_rmsprop_velocity_separable_conv2d_5_pointwise_kernel:@�O
@assignvariableop_93_rmsprop_velocity_batch_normalization_4_gamma:	�N
?assignvariableop_94_rmsprop_velocity_batch_normalization_4_beta:	�c
Hassignvariableop_95_rmsprop_velocity_separable_conv2d_4_depthwise_kernel:�d
Hassignvariableop_96_rmsprop_velocity_separable_conv2d_4_pointwise_kernel:��P
5assignvariableop_97_rmsprop_velocity_conv2d_13_kernel:@�O
@assignvariableop_98_rmsprop_velocity_batch_normalization_7_gamma:	�N
?assignvariableop_99_rmsprop_velocity_batch_normalization_7_beta:	�d
Iassignvariableop_100_rmsprop_velocity_separable_conv2d_7_depthwise_kernel:�e
Iassignvariableop_101_rmsprop_velocity_separable_conv2d_7_pointwise_kernel:��P
Aassignvariableop_102_rmsprop_velocity_batch_normalization_6_gamma:	�O
@assignvariableop_103_rmsprop_velocity_batch_normalization_6_beta:	�d
Iassignvariableop_104_rmsprop_velocity_separable_conv2d_6_depthwise_kernel:�e
Iassignvariableop_105_rmsprop_velocity_separable_conv2d_6_pointwise_kernel:��R
6assignvariableop_106_rmsprop_velocity_conv2d_14_kernel:��P
Aassignvariableop_107_rmsprop_velocity_batch_normalization_9_gamma:	�O
@assignvariableop_108_rmsprop_velocity_batch_normalization_9_beta:	�d
Iassignvariableop_109_rmsprop_velocity_separable_conv2d_9_depthwise_kernel:�e
Iassignvariableop_110_rmsprop_velocity_separable_conv2d_9_pointwise_kernel:��P
Aassignvariableop_111_rmsprop_velocity_batch_normalization_8_gamma:	�O
@assignvariableop_112_rmsprop_velocity_batch_normalization_8_beta:	�d
Iassignvariableop_113_rmsprop_velocity_separable_conv2d_8_depthwise_kernel:�e
Iassignvariableop_114_rmsprop_velocity_separable_conv2d_8_pointwise_kernel:��R
6assignvariableop_115_rmsprop_velocity_conv2d_15_kernel:��G
4assignvariableop_116_rmsprop_velocity_dense_2_kernel:	�@
2assignvariableop_117_rmsprop_velocity_dense_2_bias:&
assignvariableop_118_total_1: &
assignvariableop_119_count_1: $
assignvariableop_120_total: $
assignvariableop_121_count: 9
+assignvariableop_122_random_flip_2_statevar:	=
/assignvariableop_123_random_rotation_2_statevar:	9
+assignvariableop_124_random_zoom_2_statevar:	
identity_126��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_115�AssignVariableOp_116�AssignVariableOp_117�AssignVariableOp_118�AssignVariableOp_119�AssignVariableOp_12�AssignVariableOp_120�AssignVariableOp_121�AssignVariableOp_122�AssignVariableOp_123�AssignVariableOp_124�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:~*
dtype0*�8
value�8B�8~B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-12/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-12/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-17/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-17/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-19/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-19/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-22/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-22/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-23/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-23/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-23/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-24/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-24/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:~*
dtype0*�
value�B�~B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypes�
�2~				[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_10_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp.assignvariableop_1_batch_normalization_1_gammaIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp-assignvariableop_2_batch_normalization_1_betaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp4assignvariableop_3_batch_normalization_1_moving_meanIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp8assignvariableop_4_batch_normalization_1_moving_varianceIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp6assignvariableop_5_separable_conv2d_1_depthwise_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp6assignvariableop_6_separable_conv2d_1_pointwise_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp,assignvariableop_7_batch_normalization_gammaIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp+assignvariableop_8_batch_normalization_betaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp2assignvariableop_9_batch_normalization_moving_meanIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp7assignvariableop_10_batch_normalization_moving_varianceIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp5assignvariableop_11_separable_conv2d_depthwise_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp5assignvariableop_12_separable_conv2d_pointwise_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_11_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_3_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_3_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_3_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_3_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp7assignvariableop_18_separable_conv2d_3_depthwise_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp7assignvariableop_19_separable_conv2d_3_pointwise_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_2_gammaIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_2_betaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_2_moving_meanIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_2_moving_varianceIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp7assignvariableop_24_separable_conv2d_2_depthwise_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp7assignvariableop_25_separable_conv2d_2_pointwise_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv2d_12_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_5_gammaIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp.assignvariableop_28_batch_normalization_5_betaIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp5assignvariableop_29_batch_normalization_5_moving_meanIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp9assignvariableop_30_batch_normalization_5_moving_varianceIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp7assignvariableop_31_separable_conv2d_5_depthwise_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp7assignvariableop_32_separable_conv2d_5_pointwise_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_4_gammaIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp.assignvariableop_34_batch_normalization_4_betaIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp5assignvariableop_35_batch_normalization_4_moving_meanIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp9assignvariableop_36_batch_normalization_4_moving_varianceIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp7assignvariableop_37_separable_conv2d_4_depthwise_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp7assignvariableop_38_separable_conv2d_4_pointwise_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp$assignvariableop_39_conv2d_13_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp/assignvariableop_40_batch_normalization_7_gammaIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp.assignvariableop_41_batch_normalization_7_betaIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp5assignvariableop_42_batch_normalization_7_moving_meanIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp9assignvariableop_43_batch_normalization_7_moving_varianceIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp7assignvariableop_44_separable_conv2d_7_depthwise_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp7assignvariableop_45_separable_conv2d_7_pointwise_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp/assignvariableop_46_batch_normalization_6_gammaIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp.assignvariableop_47_batch_normalization_6_betaIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp5assignvariableop_48_batch_normalization_6_moving_meanIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp9assignvariableop_49_batch_normalization_6_moving_varianceIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp7assignvariableop_50_separable_conv2d_6_depthwise_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp7assignvariableop_51_separable_conv2d_6_pointwise_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp$assignvariableop_52_conv2d_14_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp/assignvariableop_53_batch_normalization_9_gammaIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp.assignvariableop_54_batch_normalization_9_betaIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp5assignvariableop_55_batch_normalization_9_moving_meanIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp9assignvariableop_56_batch_normalization_9_moving_varianceIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp7assignvariableop_57_separable_conv2d_9_depthwise_kernelIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp7assignvariableop_58_separable_conv2d_9_pointwise_kernelIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp/assignvariableop_59_batch_normalization_8_gammaIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp.assignvariableop_60_batch_normalization_8_betaIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp5assignvariableop_61_batch_normalization_8_moving_meanIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp9assignvariableop_62_batch_normalization_8_moving_varianceIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp7assignvariableop_63_separable_conv2d_8_depthwise_kernelIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp7assignvariableop_64_separable_conv2d_8_pointwise_kernelIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp$assignvariableop_65_conv2d_15_kernelIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp"assignvariableop_66_dense_2_kernelIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp assignvariableop_67_dense_2_biasIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_68AssignVariableOpassignvariableop_68_iterationIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp!assignvariableop_69_learning_rateIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp5assignvariableop_70_rmsprop_velocity_conv2d_10_kernelIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp@assignvariableop_71_rmsprop_velocity_batch_normalization_1_gammaIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp?assignvariableop_72_rmsprop_velocity_batch_normalization_1_betaIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOpHassignvariableop_73_rmsprop_velocity_separable_conv2d_1_depthwise_kernelIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOpHassignvariableop_74_rmsprop_velocity_separable_conv2d_1_pointwise_kernelIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp>assignvariableop_75_rmsprop_velocity_batch_normalization_gammaIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp=assignvariableop_76_rmsprop_velocity_batch_normalization_betaIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOpFassignvariableop_77_rmsprop_velocity_separable_conv2d_depthwise_kernelIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOpFassignvariableop_78_rmsprop_velocity_separable_conv2d_pointwise_kernelIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp5assignvariableop_79_rmsprop_velocity_conv2d_11_kernelIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp@assignvariableop_80_rmsprop_velocity_batch_normalization_3_gammaIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp?assignvariableop_81_rmsprop_velocity_batch_normalization_3_betaIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOpHassignvariableop_82_rmsprop_velocity_separable_conv2d_3_depthwise_kernelIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOpHassignvariableop_83_rmsprop_velocity_separable_conv2d_3_pointwise_kernelIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp@assignvariableop_84_rmsprop_velocity_batch_normalization_2_gammaIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp?assignvariableop_85_rmsprop_velocity_batch_normalization_2_betaIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOpHassignvariableop_86_rmsprop_velocity_separable_conv2d_2_depthwise_kernelIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOpHassignvariableop_87_rmsprop_velocity_separable_conv2d_2_pointwise_kernelIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp5assignvariableop_88_rmsprop_velocity_conv2d_12_kernelIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp@assignvariableop_89_rmsprop_velocity_batch_normalization_5_gammaIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp?assignvariableop_90_rmsprop_velocity_batch_normalization_5_betaIdentity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOpHassignvariableop_91_rmsprop_velocity_separable_conv2d_5_depthwise_kernelIdentity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOpHassignvariableop_92_rmsprop_velocity_separable_conv2d_5_pointwise_kernelIdentity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp@assignvariableop_93_rmsprop_velocity_batch_normalization_4_gammaIdentity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp?assignvariableop_94_rmsprop_velocity_batch_normalization_4_betaIdentity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOpHassignvariableop_95_rmsprop_velocity_separable_conv2d_4_depthwise_kernelIdentity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOpHassignvariableop_96_rmsprop_velocity_separable_conv2d_4_pointwise_kernelIdentity_96:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp5assignvariableop_97_rmsprop_velocity_conv2d_13_kernelIdentity_97:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp@assignvariableop_98_rmsprop_velocity_batch_normalization_7_gammaIdentity_98:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOp?assignvariableop_99_rmsprop_velocity_batch_normalization_7_betaIdentity_99:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOpIassignvariableop_100_rmsprop_velocity_separable_conv2d_7_depthwise_kernelIdentity_100:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOpIassignvariableop_101_rmsprop_velocity_separable_conv2d_7_pointwise_kernelIdentity_101:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOpAassignvariableop_102_rmsprop_velocity_batch_normalization_6_gammaIdentity_102:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOp@assignvariableop_103_rmsprop_velocity_batch_normalization_6_betaIdentity_103:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOpIassignvariableop_104_rmsprop_velocity_separable_conv2d_6_depthwise_kernelIdentity_104:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOpIassignvariableop_105_rmsprop_velocity_separable_conv2d_6_pointwise_kernelIdentity_105:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOp6assignvariableop_106_rmsprop_velocity_conv2d_14_kernelIdentity_106:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOpAassignvariableop_107_rmsprop_velocity_batch_normalization_9_gammaIdentity_107:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOp@assignvariableop_108_rmsprop_velocity_batch_normalization_9_betaIdentity_108:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOpIassignvariableop_109_rmsprop_velocity_separable_conv2d_9_depthwise_kernelIdentity_109:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOpIassignvariableop_110_rmsprop_velocity_separable_conv2d_9_pointwise_kernelIdentity_110:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOpAassignvariableop_111_rmsprop_velocity_batch_normalization_8_gammaIdentity_111:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOp@assignvariableop_112_rmsprop_velocity_batch_normalization_8_betaIdentity_112:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_113AssignVariableOpIassignvariableop_113_rmsprop_velocity_separable_conv2d_8_depthwise_kernelIdentity_113:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_114AssignVariableOpIassignvariableop_114_rmsprop_velocity_separable_conv2d_8_pointwise_kernelIdentity_114:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_115AssignVariableOp6assignvariableop_115_rmsprop_velocity_conv2d_15_kernelIdentity_115:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_116AssignVariableOp4assignvariableop_116_rmsprop_velocity_dense_2_kernelIdentity_116:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_117AssignVariableOp2assignvariableop_117_rmsprop_velocity_dense_2_biasIdentity_117:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_118AssignVariableOpassignvariableop_118_total_1Identity_118:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_119AssignVariableOpassignvariableop_119_count_1Identity_119:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_120AssignVariableOpassignvariableop_120_totalIdentity_120:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_121AssignVariableOpassignvariableop_121_countIdentity_121:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_122AssignVariableOp+assignvariableop_122_random_flip_2_statevarIdentity_122:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_123AssignVariableOp/assignvariableop_123_random_rotation_2_statevarIdentity_123:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_124AssignVariableOp+assignvariableop_124_random_zoom_2_statevarIdentity_124:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_125Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_126IdentityIdentity_125:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
_output_shapes
 "%
identity_126Identity_126:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_992(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:6}2
0
_user_specified_namerandom_zoom_2/StateVar::|6
4
_user_specified_namerandom_rotation_2/StateVar:6{2
0
_user_specified_namerandom_flip_2/StateVar:%z!

_user_specified_namecount:%y!

_user_specified_nametotal:'x#
!
_user_specified_name	count_1:'w#
!
_user_specified_name	total_1:=v9
7
_user_specified_nameRMSprop/velocity/dense_2/bias:?u;
9
_user_specified_name!RMSprop/velocity/dense_2/kernel:At=
;
_user_specified_name#!RMSprop/velocity/conv2d_15/kernel:TsP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_8/pointwise_kernel:TrP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_8/depthwise_kernel:KqG
E
_user_specified_name-+RMSprop/velocity/batch_normalization_8/beta:LpH
F
_user_specified_name.,RMSprop/velocity/batch_normalization_8/gamma:ToP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_9/pointwise_kernel:TnP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_9/depthwise_kernel:KmG
E
_user_specified_name-+RMSprop/velocity/batch_normalization_9/beta:LlH
F
_user_specified_name.,RMSprop/velocity/batch_normalization_9/gamma:Ak=
;
_user_specified_name#!RMSprop/velocity/conv2d_14/kernel:TjP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_6/pointwise_kernel:TiP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_6/depthwise_kernel:KhG
E
_user_specified_name-+RMSprop/velocity/batch_normalization_6/beta:LgH
F
_user_specified_name.,RMSprop/velocity/batch_normalization_6/gamma:TfP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_7/pointwise_kernel:TeP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_7/depthwise_kernel:KdG
E
_user_specified_name-+RMSprop/velocity/batch_normalization_7/beta:LcH
F
_user_specified_name.,RMSprop/velocity/batch_normalization_7/gamma:Ab=
;
_user_specified_name#!RMSprop/velocity/conv2d_13/kernel:TaP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_4/pointwise_kernel:T`P
N
_user_specified_name64RMSprop/velocity/separable_conv2d_4/depthwise_kernel:K_G
E
_user_specified_name-+RMSprop/velocity/batch_normalization_4/beta:L^H
F
_user_specified_name.,RMSprop/velocity/batch_normalization_4/gamma:T]P
N
_user_specified_name64RMSprop/velocity/separable_conv2d_5/pointwise_kernel:T\P
N
_user_specified_name64RMSprop/velocity/separable_conv2d_5/depthwise_kernel:K[G
E
_user_specified_name-+RMSprop/velocity/batch_normalization_5/beta:LZH
F
_user_specified_name.,RMSprop/velocity/batch_normalization_5/gamma:AY=
;
_user_specified_name#!RMSprop/velocity/conv2d_12/kernel:TXP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_2/pointwise_kernel:TWP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_2/depthwise_kernel:KVG
E
_user_specified_name-+RMSprop/velocity/batch_normalization_2/beta:LUH
F
_user_specified_name.,RMSprop/velocity/batch_normalization_2/gamma:TTP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_3/pointwise_kernel:TSP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_3/depthwise_kernel:KRG
E
_user_specified_name-+RMSprop/velocity/batch_normalization_3/beta:LQH
F
_user_specified_name.,RMSprop/velocity/batch_normalization_3/gamma:AP=
;
_user_specified_name#!RMSprop/velocity/conv2d_11/kernel:RON
L
_user_specified_name42RMSprop/velocity/separable_conv2d/pointwise_kernel:RNN
L
_user_specified_name42RMSprop/velocity/separable_conv2d/depthwise_kernel:IME
C
_user_specified_name+)RMSprop/velocity/batch_normalization/beta:JLF
D
_user_specified_name,*RMSprop/velocity/batch_normalization/gamma:TKP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_1/pointwise_kernel:TJP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_1/depthwise_kernel:KIG
E
_user_specified_name-+RMSprop/velocity/batch_normalization_1/beta:LHH
F
_user_specified_name.,RMSprop/velocity/batch_normalization_1/gamma:AG=
;
_user_specified_name#!RMSprop/velocity/conv2d_10/kernel:-F)
'
_user_specified_namelearning_rate:)E%
#
_user_specified_name	iteration:,D(
&
_user_specified_namedense_2/bias:.C*
(
_user_specified_namedense_2/kernel:0B,
*
_user_specified_nameconv2d_15/kernel:CA?
=
_user_specified_name%#separable_conv2d_8/pointwise_kernel:C@?
=
_user_specified_name%#separable_conv2d_8/depthwise_kernel:E?A
?
_user_specified_name'%batch_normalization_8/moving_variance:A>=
;
_user_specified_name#!batch_normalization_8/moving_mean::=6
4
_user_specified_namebatch_normalization_8/beta:;<7
5
_user_specified_namebatch_normalization_8/gamma:C;?
=
_user_specified_name%#separable_conv2d_9/pointwise_kernel:C:?
=
_user_specified_name%#separable_conv2d_9/depthwise_kernel:E9A
?
_user_specified_name'%batch_normalization_9/moving_variance:A8=
;
_user_specified_name#!batch_normalization_9/moving_mean::76
4
_user_specified_namebatch_normalization_9/beta:;67
5
_user_specified_namebatch_normalization_9/gamma:05,
*
_user_specified_nameconv2d_14/kernel:C4?
=
_user_specified_name%#separable_conv2d_6/pointwise_kernel:C3?
=
_user_specified_name%#separable_conv2d_6/depthwise_kernel:E2A
?
_user_specified_name'%batch_normalization_6/moving_variance:A1=
;
_user_specified_name#!batch_normalization_6/moving_mean::06
4
_user_specified_namebatch_normalization_6/beta:;/7
5
_user_specified_namebatch_normalization_6/gamma:C.?
=
_user_specified_name%#separable_conv2d_7/pointwise_kernel:C-?
=
_user_specified_name%#separable_conv2d_7/depthwise_kernel:E,A
?
_user_specified_name'%batch_normalization_7/moving_variance:A+=
;
_user_specified_name#!batch_normalization_7/moving_mean::*6
4
_user_specified_namebatch_normalization_7/beta:;)7
5
_user_specified_namebatch_normalization_7/gamma:0(,
*
_user_specified_nameconv2d_13/kernel:C'?
=
_user_specified_name%#separable_conv2d_4/pointwise_kernel:C&?
=
_user_specified_name%#separable_conv2d_4/depthwise_kernel:E%A
?
_user_specified_name'%batch_normalization_4/moving_variance:A$=
;
_user_specified_name#!batch_normalization_4/moving_mean::#6
4
_user_specified_namebatch_normalization_4/beta:;"7
5
_user_specified_namebatch_normalization_4/gamma:C!?
=
_user_specified_name%#separable_conv2d_5/pointwise_kernel:C ?
=
_user_specified_name%#separable_conv2d_5/depthwise_kernel:EA
?
_user_specified_name'%batch_normalization_5/moving_variance:A=
;
_user_specified_name#!batch_normalization_5/moving_mean::6
4
_user_specified_namebatch_normalization_5/beta:;7
5
_user_specified_namebatch_normalization_5/gamma:0,
*
_user_specified_nameconv2d_12/kernel:C?
=
_user_specified_name%#separable_conv2d_2/pointwise_kernel:C?
=
_user_specified_name%#separable_conv2d_2/depthwise_kernel:EA
?
_user_specified_name'%batch_normalization_2/moving_variance:A=
;
_user_specified_name#!batch_normalization_2/moving_mean::6
4
_user_specified_namebatch_normalization_2/beta:;7
5
_user_specified_namebatch_normalization_2/gamma:C?
=
_user_specified_name%#separable_conv2d_3/pointwise_kernel:C?
=
_user_specified_name%#separable_conv2d_3/depthwise_kernel:EA
?
_user_specified_name'%batch_normalization_3/moving_variance:A=
;
_user_specified_name#!batch_normalization_3/moving_mean::6
4
_user_specified_namebatch_normalization_3/beta:;7
5
_user_specified_namebatch_normalization_3/gamma:0,
*
_user_specified_nameconv2d_11/kernel:A=
;
_user_specified_name#!separable_conv2d/pointwise_kernel:A=
;
_user_specified_name#!separable_conv2d/depthwise_kernel:C?
=
_user_specified_name%#batch_normalization/moving_variance:?
;
9
_user_specified_name!batch_normalization/moving_mean:8	4
2
_user_specified_namebatch_normalization/beta:95
3
_user_specified_namebatch_normalization/gamma:C?
=
_user_specified_name%#separable_conv2d_1/pointwise_kernel:C?
=
_user_specified_name%#separable_conv2d_1/depthwise_kernel:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:A=
;
_user_specified_name#!batch_normalization_1/moving_mean::6
4
_user_specified_namebatch_normalization_1/beta:;7
5
_user_specified_namebatch_normalization_1/gamma:0,
*
_user_specified_nameconv2d_10/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
L
0__inference_max_pooling2d_9_layer_call_fn_183824

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_181716�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_183566

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_184495

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
N__inference_separable_conv2d_8_layer_call_and_return_conditional_losses_184485

inputsC
(separable_conv2d_readvariableop_resource:�F
*separable_conv2d_readvariableop_1_resource:��
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:��*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
2__inference_random_rotation_2_layer_call_fn_184660

inputs
unknown:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_random_rotation_2_layer_call_and_return_conditional_losses_181188y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184656:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
q
H__inference_sequential_2_layer_call_and_return_conditional_losses_181317
random_flip_2_input
identity�
random_flip_2/PartitionedCallPartitionedCallrandom_flip_2_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_random_flip_2_layer_call_and_return_conditional_losses_181304�
!random_rotation_2/PartitionedCallPartitionedCall&random_flip_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_random_rotation_2_layer_call_and_return_conditional_losses_181309�
random_zoom_2/PartitionedCallPartitionedCall*random_rotation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_random_zoom_2_layer_call_and_return_conditional_losses_181314x
IdentityIdentity&random_zoom_2/PartitionedCall:output:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:f b
1
_output_shapes
:�����������
-
_user_specified_namerandom_flip_2_input
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_184579

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
R
&__inference_add_4_layer_call_fn_184515
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_4_layer_call_and_return_conditional_losses_182577i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������:����������:ZV
0
_output_shapes
:����������
"
_user_specified_name
inputs_1:Z V
0
_output_shapes
:����������
"
_user_specified_name
inputs_0
�
�
N__inference_separable_conv2d_9_layer_call_and_return_conditional_losses_182147

inputsC
(separable_conv2d_readvariableop_resource:�F
*separable_conv2d_readvariableop_1_resource:��
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:��*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
3__inference_separable_conv2d_9_layer_call_fn_184380

inputs"
unknown:�%
	unknown_0:��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_9_layer_call_and_return_conditional_losses_182147�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184376:&"
 
_user_specified_name184374:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
d
H__inference_activation_9_layer_call_and_return_conditional_losses_182534

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:����������c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_184454

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
��
��
__inference__traced_save_185692
file_prefixA
'read_disablecopyonread_conv2d_10_kernel: B
4read_1_disablecopyonread_batch_normalization_1_gamma: A
3read_2_disablecopyonread_batch_normalization_1_beta: H
:read_3_disablecopyonread_batch_normalization_1_moving_mean: L
>read_4_disablecopyonread_batch_normalization_1_moving_variance: V
<read_5_disablecopyonread_separable_conv2d_1_depthwise_kernel: V
<read_6_disablecopyonread_separable_conv2d_1_pointwise_kernel:  @
2read_7_disablecopyonread_batch_normalization_gamma: ?
1read_8_disablecopyonread_batch_normalization_beta: F
8read_9_disablecopyonread_batch_normalization_moving_mean: K
=read_10_disablecopyonread_batch_normalization_moving_variance: U
;read_11_disablecopyonread_separable_conv2d_depthwise_kernel: U
;read_12_disablecopyonread_separable_conv2d_pointwise_kernel:  D
*read_13_disablecopyonread_conv2d_11_kernel:  C
5read_14_disablecopyonread_batch_normalization_3_gamma: B
4read_15_disablecopyonread_batch_normalization_3_beta: I
;read_16_disablecopyonread_batch_normalization_3_moving_mean: M
?read_17_disablecopyonread_batch_normalization_3_moving_variance: W
=read_18_disablecopyonread_separable_conv2d_3_depthwise_kernel: W
=read_19_disablecopyonread_separable_conv2d_3_pointwise_kernel: @C
5read_20_disablecopyonread_batch_normalization_2_gamma:@B
4read_21_disablecopyonread_batch_normalization_2_beta:@I
;read_22_disablecopyonread_batch_normalization_2_moving_mean:@M
?read_23_disablecopyonread_batch_normalization_2_moving_variance:@W
=read_24_disablecopyonread_separable_conv2d_2_depthwise_kernel:@W
=read_25_disablecopyonread_separable_conv2d_2_pointwise_kernel:@@D
*read_26_disablecopyonread_conv2d_12_kernel: @C
5read_27_disablecopyonread_batch_normalization_5_gamma:@B
4read_28_disablecopyonread_batch_normalization_5_beta:@I
;read_29_disablecopyonread_batch_normalization_5_moving_mean:@M
?read_30_disablecopyonread_batch_normalization_5_moving_variance:@W
=read_31_disablecopyonread_separable_conv2d_5_depthwise_kernel:@X
=read_32_disablecopyonread_separable_conv2d_5_pointwise_kernel:@�D
5read_33_disablecopyonread_batch_normalization_4_gamma:	�C
4read_34_disablecopyonread_batch_normalization_4_beta:	�J
;read_35_disablecopyonread_batch_normalization_4_moving_mean:	�N
?read_36_disablecopyonread_batch_normalization_4_moving_variance:	�X
=read_37_disablecopyonread_separable_conv2d_4_depthwise_kernel:�Y
=read_38_disablecopyonread_separable_conv2d_4_pointwise_kernel:��E
*read_39_disablecopyonread_conv2d_13_kernel:@�D
5read_40_disablecopyonread_batch_normalization_7_gamma:	�C
4read_41_disablecopyonread_batch_normalization_7_beta:	�J
;read_42_disablecopyonread_batch_normalization_7_moving_mean:	�N
?read_43_disablecopyonread_batch_normalization_7_moving_variance:	�X
=read_44_disablecopyonread_separable_conv2d_7_depthwise_kernel:�Y
=read_45_disablecopyonread_separable_conv2d_7_pointwise_kernel:��D
5read_46_disablecopyonread_batch_normalization_6_gamma:	�C
4read_47_disablecopyonread_batch_normalization_6_beta:	�J
;read_48_disablecopyonread_batch_normalization_6_moving_mean:	�N
?read_49_disablecopyonread_batch_normalization_6_moving_variance:	�X
=read_50_disablecopyonread_separable_conv2d_6_depthwise_kernel:�Y
=read_51_disablecopyonread_separable_conv2d_6_pointwise_kernel:��F
*read_52_disablecopyonread_conv2d_14_kernel:��D
5read_53_disablecopyonread_batch_normalization_9_gamma:	�C
4read_54_disablecopyonread_batch_normalization_9_beta:	�J
;read_55_disablecopyonread_batch_normalization_9_moving_mean:	�N
?read_56_disablecopyonread_batch_normalization_9_moving_variance:	�X
=read_57_disablecopyonread_separable_conv2d_9_depthwise_kernel:�Y
=read_58_disablecopyonread_separable_conv2d_9_pointwise_kernel:��D
5read_59_disablecopyonread_batch_normalization_8_gamma:	�C
4read_60_disablecopyonread_batch_normalization_8_beta:	�J
;read_61_disablecopyonread_batch_normalization_8_moving_mean:	�N
?read_62_disablecopyonread_batch_normalization_8_moving_variance:	�X
=read_63_disablecopyonread_separable_conv2d_8_depthwise_kernel:�Y
=read_64_disablecopyonread_separable_conv2d_8_pointwise_kernel:��F
*read_65_disablecopyonread_conv2d_15_kernel:��;
(read_66_disablecopyonread_dense_2_kernel:	�4
&read_67_disablecopyonread_dense_2_bias:-
#read_68_disablecopyonread_iteration:	 1
'read_69_disablecopyonread_learning_rate: U
;read_70_disablecopyonread_rmsprop_velocity_conv2d_10_kernel: T
Fread_71_disablecopyonread_rmsprop_velocity_batch_normalization_1_gamma: S
Eread_72_disablecopyonread_rmsprop_velocity_batch_normalization_1_beta: h
Nread_73_disablecopyonread_rmsprop_velocity_separable_conv2d_1_depthwise_kernel: h
Nread_74_disablecopyonread_rmsprop_velocity_separable_conv2d_1_pointwise_kernel:  R
Dread_75_disablecopyonread_rmsprop_velocity_batch_normalization_gamma: Q
Cread_76_disablecopyonread_rmsprop_velocity_batch_normalization_beta: f
Lread_77_disablecopyonread_rmsprop_velocity_separable_conv2d_depthwise_kernel: f
Lread_78_disablecopyonread_rmsprop_velocity_separable_conv2d_pointwise_kernel:  U
;read_79_disablecopyonread_rmsprop_velocity_conv2d_11_kernel:  T
Fread_80_disablecopyonread_rmsprop_velocity_batch_normalization_3_gamma: S
Eread_81_disablecopyonread_rmsprop_velocity_batch_normalization_3_beta: h
Nread_82_disablecopyonread_rmsprop_velocity_separable_conv2d_3_depthwise_kernel: h
Nread_83_disablecopyonread_rmsprop_velocity_separable_conv2d_3_pointwise_kernel: @T
Fread_84_disablecopyonread_rmsprop_velocity_batch_normalization_2_gamma:@S
Eread_85_disablecopyonread_rmsprop_velocity_batch_normalization_2_beta:@h
Nread_86_disablecopyonread_rmsprop_velocity_separable_conv2d_2_depthwise_kernel:@h
Nread_87_disablecopyonread_rmsprop_velocity_separable_conv2d_2_pointwise_kernel:@@U
;read_88_disablecopyonread_rmsprop_velocity_conv2d_12_kernel: @T
Fread_89_disablecopyonread_rmsprop_velocity_batch_normalization_5_gamma:@S
Eread_90_disablecopyonread_rmsprop_velocity_batch_normalization_5_beta:@h
Nread_91_disablecopyonread_rmsprop_velocity_separable_conv2d_5_depthwise_kernel:@i
Nread_92_disablecopyonread_rmsprop_velocity_separable_conv2d_5_pointwise_kernel:@�U
Fread_93_disablecopyonread_rmsprop_velocity_batch_normalization_4_gamma:	�T
Eread_94_disablecopyonread_rmsprop_velocity_batch_normalization_4_beta:	�i
Nread_95_disablecopyonread_rmsprop_velocity_separable_conv2d_4_depthwise_kernel:�j
Nread_96_disablecopyonread_rmsprop_velocity_separable_conv2d_4_pointwise_kernel:��V
;read_97_disablecopyonread_rmsprop_velocity_conv2d_13_kernel:@�U
Fread_98_disablecopyonread_rmsprop_velocity_batch_normalization_7_gamma:	�T
Eread_99_disablecopyonread_rmsprop_velocity_batch_normalization_7_beta:	�j
Oread_100_disablecopyonread_rmsprop_velocity_separable_conv2d_7_depthwise_kernel:�k
Oread_101_disablecopyonread_rmsprop_velocity_separable_conv2d_7_pointwise_kernel:��V
Gread_102_disablecopyonread_rmsprop_velocity_batch_normalization_6_gamma:	�U
Fread_103_disablecopyonread_rmsprop_velocity_batch_normalization_6_beta:	�j
Oread_104_disablecopyonread_rmsprop_velocity_separable_conv2d_6_depthwise_kernel:�k
Oread_105_disablecopyonread_rmsprop_velocity_separable_conv2d_6_pointwise_kernel:��X
<read_106_disablecopyonread_rmsprop_velocity_conv2d_14_kernel:��V
Gread_107_disablecopyonread_rmsprop_velocity_batch_normalization_9_gamma:	�U
Fread_108_disablecopyonread_rmsprop_velocity_batch_normalization_9_beta:	�j
Oread_109_disablecopyonread_rmsprop_velocity_separable_conv2d_9_depthwise_kernel:�k
Oread_110_disablecopyonread_rmsprop_velocity_separable_conv2d_9_pointwise_kernel:��V
Gread_111_disablecopyonread_rmsprop_velocity_batch_normalization_8_gamma:	�U
Fread_112_disablecopyonread_rmsprop_velocity_batch_normalization_8_beta:	�j
Oread_113_disablecopyonread_rmsprop_velocity_separable_conv2d_8_depthwise_kernel:�k
Oread_114_disablecopyonread_rmsprop_velocity_separable_conv2d_8_pointwise_kernel:��X
<read_115_disablecopyonread_rmsprop_velocity_conv2d_15_kernel:��M
:read_116_disablecopyonread_rmsprop_velocity_dense_2_kernel:	�F
8read_117_disablecopyonread_rmsprop_velocity_dense_2_bias:,
"read_118_disablecopyonread_total_1: ,
"read_119_disablecopyonread_count_1: *
 read_120_disablecopyonread_total: *
 read_121_disablecopyonread_count: ?
1read_122_disablecopyonread_random_flip_2_statevar:	C
5read_123_disablecopyonread_random_rotation_2_statevar:	?
1read_124_disablecopyonread_random_zoom_2_statevar:	
savev2_const
identity_251��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_100/DisableCopyOnRead�Read_100/ReadVariableOp�Read_101/DisableCopyOnRead�Read_101/ReadVariableOp�Read_102/DisableCopyOnRead�Read_102/ReadVariableOp�Read_103/DisableCopyOnRead�Read_103/ReadVariableOp�Read_104/DisableCopyOnRead�Read_104/ReadVariableOp�Read_105/DisableCopyOnRead�Read_105/ReadVariableOp�Read_106/DisableCopyOnRead�Read_106/ReadVariableOp�Read_107/DisableCopyOnRead�Read_107/ReadVariableOp�Read_108/DisableCopyOnRead�Read_108/ReadVariableOp�Read_109/DisableCopyOnRead�Read_109/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_110/DisableCopyOnRead�Read_110/ReadVariableOp�Read_111/DisableCopyOnRead�Read_111/ReadVariableOp�Read_112/DisableCopyOnRead�Read_112/ReadVariableOp�Read_113/DisableCopyOnRead�Read_113/ReadVariableOp�Read_114/DisableCopyOnRead�Read_114/ReadVariableOp�Read_115/DisableCopyOnRead�Read_115/ReadVariableOp�Read_116/DisableCopyOnRead�Read_116/ReadVariableOp�Read_117/DisableCopyOnRead�Read_117/ReadVariableOp�Read_118/DisableCopyOnRead�Read_118/ReadVariableOp�Read_119/DisableCopyOnRead�Read_119/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_120/DisableCopyOnRead�Read_120/ReadVariableOp�Read_121/DisableCopyOnRead�Read_121/ReadVariableOp�Read_122/DisableCopyOnRead�Read_122/ReadVariableOp�Read_123/DisableCopyOnRead�Read_123/ReadVariableOp�Read_124/DisableCopyOnRead�Read_124/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_72/DisableCopyOnRead�Read_72/ReadVariableOp�Read_73/DisableCopyOnRead�Read_73/ReadVariableOp�Read_74/DisableCopyOnRead�Read_74/ReadVariableOp�Read_75/DisableCopyOnRead�Read_75/ReadVariableOp�Read_76/DisableCopyOnRead�Read_76/ReadVariableOp�Read_77/DisableCopyOnRead�Read_77/ReadVariableOp�Read_78/DisableCopyOnRead�Read_78/ReadVariableOp�Read_79/DisableCopyOnRead�Read_79/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_80/DisableCopyOnRead�Read_80/ReadVariableOp�Read_81/DisableCopyOnRead�Read_81/ReadVariableOp�Read_82/DisableCopyOnRead�Read_82/ReadVariableOp�Read_83/DisableCopyOnRead�Read_83/ReadVariableOp�Read_84/DisableCopyOnRead�Read_84/ReadVariableOp�Read_85/DisableCopyOnRead�Read_85/ReadVariableOp�Read_86/DisableCopyOnRead�Read_86/ReadVariableOp�Read_87/DisableCopyOnRead�Read_87/ReadVariableOp�Read_88/DisableCopyOnRead�Read_88/ReadVariableOp�Read_89/DisableCopyOnRead�Read_89/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOp�Read_90/DisableCopyOnRead�Read_90/ReadVariableOp�Read_91/DisableCopyOnRead�Read_91/ReadVariableOp�Read_92/DisableCopyOnRead�Read_92/ReadVariableOp�Read_93/DisableCopyOnRead�Read_93/ReadVariableOp�Read_94/DisableCopyOnRead�Read_94/ReadVariableOp�Read_95/DisableCopyOnRead�Read_95/ReadVariableOp�Read_96/DisableCopyOnRead�Read_96/ReadVariableOp�Read_97/DisableCopyOnRead�Read_97/ReadVariableOp�Read_98/DisableCopyOnRead�Read_98/ReadVariableOp�Read_99/DisableCopyOnRead�Read_99/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: y
Read/DisableCopyOnReadDisableCopyOnRead'read_disablecopyonread_conv2d_10_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp'read_disablecopyonread_conv2d_10_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_1/DisableCopyOnReadDisableCopyOnRead4read_1_disablecopyonread_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp4read_1_disablecopyonread_batch_normalization_1_gamma^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_2/DisableCopyOnReadDisableCopyOnRead3read_2_disablecopyonread_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp3read_2_disablecopyonread_batch_normalization_1_beta^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_3/DisableCopyOnReadDisableCopyOnRead:read_3_disablecopyonread_batch_normalization_1_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp:read_3_disablecopyonread_batch_normalization_1_moving_mean^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_4/DisableCopyOnReadDisableCopyOnRead>read_4_disablecopyonread_batch_normalization_1_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp>read_4_disablecopyonread_batch_normalization_1_moving_variance^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_5/DisableCopyOnReadDisableCopyOnRead<read_5_disablecopyonread_separable_conv2d_1_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp<read_5_disablecopyonread_separable_conv2d_1_depthwise_kernel^Read_5/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0v
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_6/DisableCopyOnReadDisableCopyOnRead<read_6_disablecopyonread_separable_conv2d_1_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp<read_6_disablecopyonread_separable_conv2d_1_pointwise_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:  �
Read_7/DisableCopyOnReadDisableCopyOnRead2read_7_disablecopyonread_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp2read_7_disablecopyonread_batch_normalization_gamma^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_8/DisableCopyOnReadDisableCopyOnRead1read_8_disablecopyonread_batch_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp1read_8_disablecopyonread_batch_normalization_beta^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_9/DisableCopyOnReadDisableCopyOnRead8read_9_disablecopyonread_batch_normalization_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp8read_9_disablecopyonread_batch_normalization_moving_mean^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_10/DisableCopyOnReadDisableCopyOnRead=read_10_disablecopyonread_batch_normalization_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp=read_10_disablecopyonread_batch_normalization_moving_variance^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_11/DisableCopyOnReadDisableCopyOnRead;read_11_disablecopyonread_separable_conv2d_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp;read_11_disablecopyonread_separable_conv2d_depthwise_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_12/DisableCopyOnReadDisableCopyOnRead;read_12_disablecopyonread_separable_conv2d_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp;read_12_disablecopyonread_separable_conv2d_pointwise_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0w
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  m
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*&
_output_shapes
:  
Read_13/DisableCopyOnReadDisableCopyOnRead*read_13_disablecopyonread_conv2d_11_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp*read_13_disablecopyonread_conv2d_11_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0w
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  m
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*&
_output_shapes
:  �
Read_14/DisableCopyOnReadDisableCopyOnRead5read_14_disablecopyonread_batch_normalization_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp5read_14_disablecopyonread_batch_normalization_3_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_15/DisableCopyOnReadDisableCopyOnRead4read_15_disablecopyonread_batch_normalization_3_beta"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp4read_15_disablecopyonread_batch_normalization_3_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_16/DisableCopyOnReadDisableCopyOnRead;read_16_disablecopyonread_batch_normalization_3_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp;read_16_disablecopyonread_batch_normalization_3_moving_mean^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_17/DisableCopyOnReadDisableCopyOnRead?read_17_disablecopyonread_batch_normalization_3_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp?read_17_disablecopyonread_batch_normalization_3_moving_variance^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_18/DisableCopyOnReadDisableCopyOnRead=read_18_disablecopyonread_separable_conv2d_3_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp=read_18_disablecopyonread_separable_conv2d_3_depthwise_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0w
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: m
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_19/DisableCopyOnReadDisableCopyOnRead=read_19_disablecopyonread_separable_conv2d_3_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp=read_19_disablecopyonread_separable_conv2d_3_pointwise_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*&
_output_shapes
: @�
Read_20/DisableCopyOnReadDisableCopyOnRead5read_20_disablecopyonread_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp5read_20_disablecopyonread_batch_normalization_2_gamma^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_21/DisableCopyOnReadDisableCopyOnRead4read_21_disablecopyonread_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp4read_21_disablecopyonread_batch_normalization_2_beta^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_22/DisableCopyOnReadDisableCopyOnRead;read_22_disablecopyonread_batch_normalization_2_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp;read_22_disablecopyonread_batch_normalization_2_moving_mean^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_23/DisableCopyOnReadDisableCopyOnRead?read_23_disablecopyonread_batch_normalization_2_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp?read_23_disablecopyonread_batch_normalization_2_moving_variance^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_24/DisableCopyOnReadDisableCopyOnRead=read_24_disablecopyonread_separable_conv2d_2_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp=read_24_disablecopyonread_separable_conv2d_2_depthwise_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_25/DisableCopyOnReadDisableCopyOnRead=read_25_disablecopyonread_separable_conv2d_2_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp=read_25_disablecopyonread_separable_conv2d_2_pointwise_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@
Read_26/DisableCopyOnReadDisableCopyOnRead*read_26_disablecopyonread_conv2d_12_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp*read_26_disablecopyonread_conv2d_12_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0w
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @m
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*&
_output_shapes
: @�
Read_27/DisableCopyOnReadDisableCopyOnRead5read_27_disablecopyonread_batch_normalization_5_gamma"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp5read_27_disablecopyonread_batch_normalization_5_gamma^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_28/DisableCopyOnReadDisableCopyOnRead4read_28_disablecopyonread_batch_normalization_5_beta"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp4read_28_disablecopyonread_batch_normalization_5_beta^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_29/DisableCopyOnReadDisableCopyOnRead;read_29_disablecopyonread_batch_normalization_5_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp;read_29_disablecopyonread_batch_normalization_5_moving_mean^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_30/DisableCopyOnReadDisableCopyOnRead?read_30_disablecopyonread_batch_normalization_5_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp?read_30_disablecopyonread_batch_normalization_5_moving_variance^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_31/DisableCopyOnReadDisableCopyOnRead=read_31_disablecopyonread_separable_conv2d_5_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp=read_31_disablecopyonread_separable_conv2d_5_depthwise_kernel^Read_31/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_32/DisableCopyOnReadDisableCopyOnRead=read_32_disablecopyonread_separable_conv2d_5_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp=read_32_disablecopyonread_separable_conv2d_5_pointwise_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_33/DisableCopyOnReadDisableCopyOnRead5read_33_disablecopyonread_batch_normalization_4_gamma"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp5read_33_disablecopyonread_batch_normalization_4_gamma^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_34/DisableCopyOnReadDisableCopyOnRead4read_34_disablecopyonread_batch_normalization_4_beta"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp4read_34_disablecopyonread_batch_normalization_4_beta^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_35/DisableCopyOnReadDisableCopyOnRead;read_35_disablecopyonread_batch_normalization_4_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp;read_35_disablecopyonread_batch_normalization_4_moving_mean^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_36/DisableCopyOnReadDisableCopyOnRead?read_36_disablecopyonread_batch_normalization_4_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp?read_36_disablecopyonread_batch_normalization_4_moving_variance^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_37/DisableCopyOnReadDisableCopyOnRead=read_37_disablecopyonread_separable_conv2d_4_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp=read_37_disablecopyonread_separable_conv2d_4_depthwise_kernel^Read_37/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�*
dtype0x
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�n
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*'
_output_shapes
:��
Read_38/DisableCopyOnReadDisableCopyOnRead=read_38_disablecopyonread_separable_conv2d_4_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp=read_38_disablecopyonread_separable_conv2d_4_pointwise_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*(
_output_shapes
:��
Read_39/DisableCopyOnReadDisableCopyOnRead*read_39_disablecopyonread_conv2d_13_kernel"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp*read_39_disablecopyonread_conv2d_13_kernel^Read_39/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_40/DisableCopyOnReadDisableCopyOnRead5read_40_disablecopyonread_batch_normalization_7_gamma"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp5read_40_disablecopyonread_batch_normalization_7_gamma^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_41/DisableCopyOnReadDisableCopyOnRead4read_41_disablecopyonread_batch_normalization_7_beta"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp4read_41_disablecopyonread_batch_normalization_7_beta^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_42/DisableCopyOnReadDisableCopyOnRead;read_42_disablecopyonread_batch_normalization_7_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp;read_42_disablecopyonread_batch_normalization_7_moving_mean^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_43/DisableCopyOnReadDisableCopyOnRead?read_43_disablecopyonread_batch_normalization_7_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp?read_43_disablecopyonread_batch_normalization_7_moving_variance^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_44/DisableCopyOnReadDisableCopyOnRead=read_44_disablecopyonread_separable_conv2d_7_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp=read_44_disablecopyonread_separable_conv2d_7_depthwise_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�*
dtype0x
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�n
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*'
_output_shapes
:��
Read_45/DisableCopyOnReadDisableCopyOnRead=read_45_disablecopyonread_separable_conv2d_7_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp=read_45_disablecopyonread_separable_conv2d_7_pointwise_kernel^Read_45/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_46/DisableCopyOnReadDisableCopyOnRead5read_46_disablecopyonread_batch_normalization_6_gamma"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp5read_46_disablecopyonread_batch_normalization_6_gamma^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_47/DisableCopyOnReadDisableCopyOnRead4read_47_disablecopyonread_batch_normalization_6_beta"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp4read_47_disablecopyonread_batch_normalization_6_beta^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_48/DisableCopyOnReadDisableCopyOnRead;read_48_disablecopyonread_batch_normalization_6_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp;read_48_disablecopyonread_batch_normalization_6_moving_mean^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_49/DisableCopyOnReadDisableCopyOnRead?read_49_disablecopyonread_batch_normalization_6_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp?read_49_disablecopyonread_batch_normalization_6_moving_variance^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_50/DisableCopyOnReadDisableCopyOnRead=read_50_disablecopyonread_separable_conv2d_6_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp=read_50_disablecopyonread_separable_conv2d_6_depthwise_kernel^Read_50/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�*
dtype0y
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�p
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*'
_output_shapes
:��
Read_51/DisableCopyOnReadDisableCopyOnRead=read_51_disablecopyonread_separable_conv2d_6_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp=read_51_disablecopyonread_separable_conv2d_6_pointwise_kernel^Read_51/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*(
_output_shapes
:��
Read_52/DisableCopyOnReadDisableCopyOnRead*read_52_disablecopyonread_conv2d_14_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp*read_52_disablecopyonread_conv2d_14_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_53/DisableCopyOnReadDisableCopyOnRead5read_53_disablecopyonread_batch_normalization_9_gamma"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp5read_53_disablecopyonread_batch_normalization_9_gamma^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_54/DisableCopyOnReadDisableCopyOnRead4read_54_disablecopyonread_batch_normalization_9_beta"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp4read_54_disablecopyonread_batch_normalization_9_beta^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_55/DisableCopyOnReadDisableCopyOnRead;read_55_disablecopyonread_batch_normalization_9_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp;read_55_disablecopyonread_batch_normalization_9_moving_mean^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_56/DisableCopyOnReadDisableCopyOnRead?read_56_disablecopyonread_batch_normalization_9_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp?read_56_disablecopyonread_batch_normalization_9_moving_variance^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_57/DisableCopyOnReadDisableCopyOnRead=read_57_disablecopyonread_separable_conv2d_9_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp=read_57_disablecopyonread_separable_conv2d_9_depthwise_kernel^Read_57/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�*
dtype0y
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�p
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*'
_output_shapes
:��
Read_58/DisableCopyOnReadDisableCopyOnRead=read_58_disablecopyonread_separable_conv2d_9_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp=read_58_disablecopyonread_separable_conv2d_9_pointwise_kernel^Read_58/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_59/DisableCopyOnReadDisableCopyOnRead5read_59_disablecopyonread_batch_normalization_8_gamma"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp5read_59_disablecopyonread_batch_normalization_8_gamma^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_60/DisableCopyOnReadDisableCopyOnRead4read_60_disablecopyonread_batch_normalization_8_beta"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp4read_60_disablecopyonread_batch_normalization_8_beta^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_61/DisableCopyOnReadDisableCopyOnRead;read_61_disablecopyonread_batch_normalization_8_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp;read_61_disablecopyonread_batch_normalization_8_moving_mean^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_62/DisableCopyOnReadDisableCopyOnRead?read_62_disablecopyonread_batch_normalization_8_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp?read_62_disablecopyonread_batch_normalization_8_moving_variance^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_63/DisableCopyOnReadDisableCopyOnRead=read_63_disablecopyonread_separable_conv2d_8_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp=read_63_disablecopyonread_separable_conv2d_8_depthwise_kernel^Read_63/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�*
dtype0y
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�p
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*'
_output_shapes
:��
Read_64/DisableCopyOnReadDisableCopyOnRead=read_64_disablecopyonread_separable_conv2d_8_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp=read_64_disablecopyonread_separable_conv2d_8_pointwise_kernel^Read_64/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*(
_output_shapes
:��
Read_65/DisableCopyOnReadDisableCopyOnRead*read_65_disablecopyonread_conv2d_15_kernel"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp*read_65_disablecopyonread_conv2d_15_kernel^Read_65/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*(
_output_shapes
:��}
Read_66/DisableCopyOnReadDisableCopyOnRead(read_66_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp(read_66_disablecopyonread_dense_2_kernel^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
:	�{
Read_67/DisableCopyOnReadDisableCopyOnRead&read_67_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp&read_67_disablecopyonread_dense_2_bias^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_68/DisableCopyOnReadDisableCopyOnRead#read_68_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOp#read_68_disablecopyonread_iteration^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	h
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: _
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_69/DisableCopyOnReadDisableCopyOnRead'read_69_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp'read_69_disablecopyonread_learning_rate^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_70/DisableCopyOnReadDisableCopyOnRead;read_70_disablecopyonread_rmsprop_velocity_conv2d_10_kernel"/device:CPU:0*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOp;read_70_disablecopyonread_rmsprop_velocity_conv2d_10_kernel^Read_70/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0x
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_71/DisableCopyOnReadDisableCopyOnReadFread_71_disablecopyonread_rmsprop_velocity_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOpFread_71_disablecopyonread_rmsprop_velocity_batch_normalization_1_gamma^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_72/DisableCopyOnReadDisableCopyOnReadEread_72_disablecopyonread_rmsprop_velocity_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_72/ReadVariableOpReadVariableOpEread_72_disablecopyonread_rmsprop_velocity_batch_normalization_1_beta^Read_72/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_73/DisableCopyOnReadDisableCopyOnReadNread_73_disablecopyonread_rmsprop_velocity_separable_conv2d_1_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_73/ReadVariableOpReadVariableOpNread_73_disablecopyonread_rmsprop_velocity_separable_conv2d_1_depthwise_kernel^Read_73/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0x
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_74/DisableCopyOnReadDisableCopyOnReadNread_74_disablecopyonread_rmsprop_velocity_separable_conv2d_1_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_74/ReadVariableOpReadVariableOpNread_74_disablecopyonread_rmsprop_velocity_separable_conv2d_1_pointwise_kernel^Read_74/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0x
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  o
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*&
_output_shapes
:  �
Read_75/DisableCopyOnReadDisableCopyOnReadDread_75_disablecopyonread_rmsprop_velocity_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_75/ReadVariableOpReadVariableOpDread_75_disablecopyonread_rmsprop_velocity_batch_normalization_gamma^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_76/DisableCopyOnReadDisableCopyOnReadCread_76_disablecopyonread_rmsprop_velocity_batch_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_76/ReadVariableOpReadVariableOpCread_76_disablecopyonread_rmsprop_velocity_batch_normalization_beta^Read_76/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_77/DisableCopyOnReadDisableCopyOnReadLread_77_disablecopyonread_rmsprop_velocity_separable_conv2d_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_77/ReadVariableOpReadVariableOpLread_77_disablecopyonread_rmsprop_velocity_separable_conv2d_depthwise_kernel^Read_77/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0x
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_78/DisableCopyOnReadDisableCopyOnReadLread_78_disablecopyonread_rmsprop_velocity_separable_conv2d_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_78/ReadVariableOpReadVariableOpLread_78_disablecopyonread_rmsprop_velocity_separable_conv2d_pointwise_kernel^Read_78/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0x
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  o
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*&
_output_shapes
:  �
Read_79/DisableCopyOnReadDisableCopyOnRead;read_79_disablecopyonread_rmsprop_velocity_conv2d_11_kernel"/device:CPU:0*
_output_shapes
 �
Read_79/ReadVariableOpReadVariableOp;read_79_disablecopyonread_rmsprop_velocity_conv2d_11_kernel^Read_79/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:  *
dtype0x
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:  o
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*&
_output_shapes
:  �
Read_80/DisableCopyOnReadDisableCopyOnReadFread_80_disablecopyonread_rmsprop_velocity_batch_normalization_3_gamma"/device:CPU:0*
_output_shapes
 �
Read_80/ReadVariableOpReadVariableOpFread_80_disablecopyonread_rmsprop_velocity_batch_normalization_3_gamma^Read_80/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_81/DisableCopyOnReadDisableCopyOnReadEread_81_disablecopyonread_rmsprop_velocity_batch_normalization_3_beta"/device:CPU:0*
_output_shapes
 �
Read_81/ReadVariableOpReadVariableOpEread_81_disablecopyonread_rmsprop_velocity_batch_normalization_3_beta^Read_81/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_82/DisableCopyOnReadDisableCopyOnReadNread_82_disablecopyonread_rmsprop_velocity_separable_conv2d_3_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_82/ReadVariableOpReadVariableOpNread_82_disablecopyonread_rmsprop_velocity_separable_conv2d_3_depthwise_kernel^Read_82/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: *
dtype0x
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: o
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0*&
_output_shapes
: �
Read_83/DisableCopyOnReadDisableCopyOnReadNread_83_disablecopyonread_rmsprop_velocity_separable_conv2d_3_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_83/ReadVariableOpReadVariableOpNread_83_disablecopyonread_rmsprop_velocity_separable_conv2d_3_pointwise_kernel^Read_83/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0x
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @o
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*&
_output_shapes
: @�
Read_84/DisableCopyOnReadDisableCopyOnReadFread_84_disablecopyonread_rmsprop_velocity_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_84/ReadVariableOpReadVariableOpFread_84_disablecopyonread_rmsprop_velocity_batch_normalization_2_gamma^Read_84/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_85/DisableCopyOnReadDisableCopyOnReadEread_85_disablecopyonread_rmsprop_velocity_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 �
Read_85/ReadVariableOpReadVariableOpEread_85_disablecopyonread_rmsprop_velocity_batch_normalization_2_beta^Read_85/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_86/DisableCopyOnReadDisableCopyOnReadNread_86_disablecopyonread_rmsprop_velocity_separable_conv2d_2_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_86/ReadVariableOpReadVariableOpNread_86_disablecopyonread_rmsprop_velocity_separable_conv2d_2_depthwise_kernel^Read_86/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0x
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@o
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_87/DisableCopyOnReadDisableCopyOnReadNread_87_disablecopyonread_rmsprop_velocity_separable_conv2d_2_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_87/ReadVariableOpReadVariableOpNread_87_disablecopyonread_rmsprop_velocity_separable_conv2d_2_pointwise_kernel^Read_87/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0x
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@�
Read_88/DisableCopyOnReadDisableCopyOnRead;read_88_disablecopyonread_rmsprop_velocity_conv2d_12_kernel"/device:CPU:0*
_output_shapes
 �
Read_88/ReadVariableOpReadVariableOp;read_88_disablecopyonread_rmsprop_velocity_conv2d_12_kernel^Read_88/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
: @*
dtype0x
Identity_176IdentityRead_88/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
: @o
Identity_177IdentityIdentity_176:output:0"/device:CPU:0*
T0*&
_output_shapes
: @�
Read_89/DisableCopyOnReadDisableCopyOnReadFread_89_disablecopyonread_rmsprop_velocity_batch_normalization_5_gamma"/device:CPU:0*
_output_shapes
 �
Read_89/ReadVariableOpReadVariableOpFread_89_disablecopyonread_rmsprop_velocity_batch_normalization_5_gamma^Read_89/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_178IdentityRead_89/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_179IdentityIdentity_178:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_90/DisableCopyOnReadDisableCopyOnReadEread_90_disablecopyonread_rmsprop_velocity_batch_normalization_5_beta"/device:CPU:0*
_output_shapes
 �
Read_90/ReadVariableOpReadVariableOpEread_90_disablecopyonread_rmsprop_velocity_batch_normalization_5_beta^Read_90/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_180IdentityRead_90/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_181IdentityIdentity_180:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_91/DisableCopyOnReadDisableCopyOnReadNread_91_disablecopyonread_rmsprop_velocity_separable_conv2d_5_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_91/ReadVariableOpReadVariableOpNread_91_disablecopyonread_rmsprop_velocity_separable_conv2d_5_depthwise_kernel^Read_91/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0x
Identity_182IdentityRead_91/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@o
Identity_183IdentityIdentity_182:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_92/DisableCopyOnReadDisableCopyOnReadNread_92_disablecopyonread_rmsprop_velocity_separable_conv2d_5_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_92/ReadVariableOpReadVariableOpNread_92_disablecopyonread_rmsprop_velocity_separable_conv2d_5_pointwise_kernel^Read_92/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0y
Identity_184IdentityRead_92/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�p
Identity_185IdentityIdentity_184:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_93/DisableCopyOnReadDisableCopyOnReadFread_93_disablecopyonread_rmsprop_velocity_batch_normalization_4_gamma"/device:CPU:0*
_output_shapes
 �
Read_93/ReadVariableOpReadVariableOpFread_93_disablecopyonread_rmsprop_velocity_batch_normalization_4_gamma^Read_93/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_186IdentityRead_93/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_187IdentityIdentity_186:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_94/DisableCopyOnReadDisableCopyOnReadEread_94_disablecopyonread_rmsprop_velocity_batch_normalization_4_beta"/device:CPU:0*
_output_shapes
 �
Read_94/ReadVariableOpReadVariableOpEread_94_disablecopyonread_rmsprop_velocity_batch_normalization_4_beta^Read_94/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_188IdentityRead_94/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_189IdentityIdentity_188:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_95/DisableCopyOnReadDisableCopyOnReadNread_95_disablecopyonread_rmsprop_velocity_separable_conv2d_4_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_95/ReadVariableOpReadVariableOpNread_95_disablecopyonread_rmsprop_velocity_separable_conv2d_4_depthwise_kernel^Read_95/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�*
dtype0y
Identity_190IdentityRead_95/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�p
Identity_191IdentityIdentity_190:output:0"/device:CPU:0*
T0*'
_output_shapes
:��
Read_96/DisableCopyOnReadDisableCopyOnReadNread_96_disablecopyonread_rmsprop_velocity_separable_conv2d_4_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_96/ReadVariableOpReadVariableOpNread_96_disablecopyonread_rmsprop_velocity_separable_conv2d_4_pointwise_kernel^Read_96/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_192IdentityRead_96/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_193IdentityIdentity_192:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_97/DisableCopyOnReadDisableCopyOnRead;read_97_disablecopyonread_rmsprop_velocity_conv2d_13_kernel"/device:CPU:0*
_output_shapes
 �
Read_97/ReadVariableOpReadVariableOp;read_97_disablecopyonread_rmsprop_velocity_conv2d_13_kernel^Read_97/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0y
Identity_194IdentityRead_97/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�p
Identity_195IdentityIdentity_194:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_98/DisableCopyOnReadDisableCopyOnReadFread_98_disablecopyonread_rmsprop_velocity_batch_normalization_7_gamma"/device:CPU:0*
_output_shapes
 �
Read_98/ReadVariableOpReadVariableOpFread_98_disablecopyonread_rmsprop_velocity_batch_normalization_7_gamma^Read_98/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_196IdentityRead_98/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_197IdentityIdentity_196:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_99/DisableCopyOnReadDisableCopyOnReadEread_99_disablecopyonread_rmsprop_velocity_batch_normalization_7_beta"/device:CPU:0*
_output_shapes
 �
Read_99/ReadVariableOpReadVariableOpEread_99_disablecopyonread_rmsprop_velocity_batch_normalization_7_beta^Read_99/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_198IdentityRead_99/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_199IdentityIdentity_198:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_100/DisableCopyOnReadDisableCopyOnReadOread_100_disablecopyonread_rmsprop_velocity_separable_conv2d_7_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_100/ReadVariableOpReadVariableOpOread_100_disablecopyonread_rmsprop_velocity_separable_conv2d_7_depthwise_kernel^Read_100/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�*
dtype0z
Identity_200IdentityRead_100/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�p
Identity_201IdentityIdentity_200:output:0"/device:CPU:0*
T0*'
_output_shapes
:��
Read_101/DisableCopyOnReadDisableCopyOnReadOread_101_disablecopyonread_rmsprop_velocity_separable_conv2d_7_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_101/ReadVariableOpReadVariableOpOread_101_disablecopyonread_rmsprop_velocity_separable_conv2d_7_pointwise_kernel^Read_101/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0{
Identity_202IdentityRead_101/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_203IdentityIdentity_202:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_102/DisableCopyOnReadDisableCopyOnReadGread_102_disablecopyonread_rmsprop_velocity_batch_normalization_6_gamma"/device:CPU:0*
_output_shapes
 �
Read_102/ReadVariableOpReadVariableOpGread_102_disablecopyonread_rmsprop_velocity_batch_normalization_6_gamma^Read_102/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0n
Identity_204IdentityRead_102/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_205IdentityIdentity_204:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_103/DisableCopyOnReadDisableCopyOnReadFread_103_disablecopyonread_rmsprop_velocity_batch_normalization_6_beta"/device:CPU:0*
_output_shapes
 �
Read_103/ReadVariableOpReadVariableOpFread_103_disablecopyonread_rmsprop_velocity_batch_normalization_6_beta^Read_103/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0n
Identity_206IdentityRead_103/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_207IdentityIdentity_206:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_104/DisableCopyOnReadDisableCopyOnReadOread_104_disablecopyonread_rmsprop_velocity_separable_conv2d_6_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_104/ReadVariableOpReadVariableOpOread_104_disablecopyonread_rmsprop_velocity_separable_conv2d_6_depthwise_kernel^Read_104/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�*
dtype0z
Identity_208IdentityRead_104/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�p
Identity_209IdentityIdentity_208:output:0"/device:CPU:0*
T0*'
_output_shapes
:��
Read_105/DisableCopyOnReadDisableCopyOnReadOread_105_disablecopyonread_rmsprop_velocity_separable_conv2d_6_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_105/ReadVariableOpReadVariableOpOread_105_disablecopyonread_rmsprop_velocity_separable_conv2d_6_pointwise_kernel^Read_105/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0{
Identity_210IdentityRead_105/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_211IdentityIdentity_210:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_106/DisableCopyOnReadDisableCopyOnRead<read_106_disablecopyonread_rmsprop_velocity_conv2d_14_kernel"/device:CPU:0*
_output_shapes
 �
Read_106/ReadVariableOpReadVariableOp<read_106_disablecopyonread_rmsprop_velocity_conv2d_14_kernel^Read_106/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0{
Identity_212IdentityRead_106/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_213IdentityIdentity_212:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_107/DisableCopyOnReadDisableCopyOnReadGread_107_disablecopyonread_rmsprop_velocity_batch_normalization_9_gamma"/device:CPU:0*
_output_shapes
 �
Read_107/ReadVariableOpReadVariableOpGread_107_disablecopyonread_rmsprop_velocity_batch_normalization_9_gamma^Read_107/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0n
Identity_214IdentityRead_107/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_215IdentityIdentity_214:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_108/DisableCopyOnReadDisableCopyOnReadFread_108_disablecopyonread_rmsprop_velocity_batch_normalization_9_beta"/device:CPU:0*
_output_shapes
 �
Read_108/ReadVariableOpReadVariableOpFread_108_disablecopyonread_rmsprop_velocity_batch_normalization_9_beta^Read_108/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0n
Identity_216IdentityRead_108/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_217IdentityIdentity_216:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_109/DisableCopyOnReadDisableCopyOnReadOread_109_disablecopyonread_rmsprop_velocity_separable_conv2d_9_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_109/ReadVariableOpReadVariableOpOread_109_disablecopyonread_rmsprop_velocity_separable_conv2d_9_depthwise_kernel^Read_109/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�*
dtype0z
Identity_218IdentityRead_109/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�p
Identity_219IdentityIdentity_218:output:0"/device:CPU:0*
T0*'
_output_shapes
:��
Read_110/DisableCopyOnReadDisableCopyOnReadOread_110_disablecopyonread_rmsprop_velocity_separable_conv2d_9_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_110/ReadVariableOpReadVariableOpOread_110_disablecopyonread_rmsprop_velocity_separable_conv2d_9_pointwise_kernel^Read_110/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0{
Identity_220IdentityRead_110/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_221IdentityIdentity_220:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_111/DisableCopyOnReadDisableCopyOnReadGread_111_disablecopyonread_rmsprop_velocity_batch_normalization_8_gamma"/device:CPU:0*
_output_shapes
 �
Read_111/ReadVariableOpReadVariableOpGread_111_disablecopyonread_rmsprop_velocity_batch_normalization_8_gamma^Read_111/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0n
Identity_222IdentityRead_111/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_223IdentityIdentity_222:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_112/DisableCopyOnReadDisableCopyOnReadFread_112_disablecopyonread_rmsprop_velocity_batch_normalization_8_beta"/device:CPU:0*
_output_shapes
 �
Read_112/ReadVariableOpReadVariableOpFread_112_disablecopyonread_rmsprop_velocity_batch_normalization_8_beta^Read_112/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0n
Identity_224IdentityRead_112/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_225IdentityIdentity_224:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_113/DisableCopyOnReadDisableCopyOnReadOread_113_disablecopyonread_rmsprop_velocity_separable_conv2d_8_depthwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_113/ReadVariableOpReadVariableOpOread_113_disablecopyonread_rmsprop_velocity_separable_conv2d_8_depthwise_kernel^Read_113/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:�*
dtype0z
Identity_226IdentityRead_113/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:�p
Identity_227IdentityIdentity_226:output:0"/device:CPU:0*
T0*'
_output_shapes
:��
Read_114/DisableCopyOnReadDisableCopyOnReadOread_114_disablecopyonread_rmsprop_velocity_separable_conv2d_8_pointwise_kernel"/device:CPU:0*
_output_shapes
 �
Read_114/ReadVariableOpReadVariableOpOread_114_disablecopyonread_rmsprop_velocity_separable_conv2d_8_pointwise_kernel^Read_114/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0{
Identity_228IdentityRead_114/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_229IdentityIdentity_228:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_115/DisableCopyOnReadDisableCopyOnRead<read_115_disablecopyonread_rmsprop_velocity_conv2d_15_kernel"/device:CPU:0*
_output_shapes
 �
Read_115/ReadVariableOpReadVariableOp<read_115_disablecopyonread_rmsprop_velocity_conv2d_15_kernel^Read_115/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0{
Identity_230IdentityRead_115/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_231IdentityIdentity_230:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_116/DisableCopyOnReadDisableCopyOnRead:read_116_disablecopyonread_rmsprop_velocity_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_116/ReadVariableOpReadVariableOp:read_116_disablecopyonread_rmsprop_velocity_dense_2_kernel^Read_116/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0r
Identity_232IdentityRead_116/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_233IdentityIdentity_232:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_117/DisableCopyOnReadDisableCopyOnRead8read_117_disablecopyonread_rmsprop_velocity_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_117/ReadVariableOpReadVariableOp8read_117_disablecopyonread_rmsprop_velocity_dense_2_bias^Read_117/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_234IdentityRead_117/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_235IdentityIdentity_234:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_118/DisableCopyOnReadDisableCopyOnRead"read_118_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_118/ReadVariableOpReadVariableOp"read_118_disablecopyonread_total_1^Read_118/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_236IdentityRead_118/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_237IdentityIdentity_236:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_119/DisableCopyOnReadDisableCopyOnRead"read_119_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_119/ReadVariableOpReadVariableOp"read_119_disablecopyonread_count_1^Read_119/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_238IdentityRead_119/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_239IdentityIdentity_238:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_120/DisableCopyOnReadDisableCopyOnRead read_120_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_120/ReadVariableOpReadVariableOp read_120_disablecopyonread_total^Read_120/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_240IdentityRead_120/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_241IdentityIdentity_240:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_121/DisableCopyOnReadDisableCopyOnRead read_121_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_121/ReadVariableOpReadVariableOp read_121_disablecopyonread_count^Read_121/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_242IdentityRead_121/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_243IdentityIdentity_242:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_122/DisableCopyOnReadDisableCopyOnRead1read_122_disablecopyonread_random_flip_2_statevar"/device:CPU:0*
_output_shapes
 �
Read_122/ReadVariableOpReadVariableOp1read_122_disablecopyonread_random_flip_2_statevar^Read_122/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0	m
Identity_244IdentityRead_122/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
:c
Identity_245IdentityIdentity_244:output:0"/device:CPU:0*
T0	*
_output_shapes
:�
Read_123/DisableCopyOnReadDisableCopyOnRead5read_123_disablecopyonread_random_rotation_2_statevar"/device:CPU:0*
_output_shapes
 �
Read_123/ReadVariableOpReadVariableOp5read_123_disablecopyonread_random_rotation_2_statevar^Read_123/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0	m
Identity_246IdentityRead_123/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
:c
Identity_247IdentityIdentity_246:output:0"/device:CPU:0*
T0	*
_output_shapes
:�
Read_124/DisableCopyOnReadDisableCopyOnRead1read_124_disablecopyonread_random_zoom_2_statevar"/device:CPU:0*
_output_shapes
 �
Read_124/ReadVariableOpReadVariableOp1read_124_disablecopyonread_random_zoom_2_statevar^Read_124/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0	m
Identity_248IdentityRead_124/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
:c
Identity_249IdentityIdentity_248:output:0"/device:CPU:0*
T0	*
_output_shapes
:�9
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:~*
dtype0*�8
value�8B�8~B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-4/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-9/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-12/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-12/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-17/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-17/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-19/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-19/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-22/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-22/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-23/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-23/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-23/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-24/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-24/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-25/kernel/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-26/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-26/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-1/layer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:~*
dtype0*�
value�B�~B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0Identity_177:output:0Identity_179:output:0Identity_181:output:0Identity_183:output:0Identity_185:output:0Identity_187:output:0Identity_189:output:0Identity_191:output:0Identity_193:output:0Identity_195:output:0Identity_197:output:0Identity_199:output:0Identity_201:output:0Identity_203:output:0Identity_205:output:0Identity_207:output:0Identity_209:output:0Identity_211:output:0Identity_213:output:0Identity_215:output:0Identity_217:output:0Identity_219:output:0Identity_221:output:0Identity_223:output:0Identity_225:output:0Identity_227:output:0Identity_229:output:0Identity_231:output:0Identity_233:output:0Identity_235:output:0Identity_237:output:0Identity_239:output:0Identity_241:output:0Identity_243:output:0Identity_245:output:0Identity_247:output:0Identity_249:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *�
dtypes�
�2~				�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_250Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_251IdentityIdentity_250:output:0^NoOp*
T0*
_output_shapes
: �4
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_100/DisableCopyOnRead^Read_100/ReadVariableOp^Read_101/DisableCopyOnRead^Read_101/ReadVariableOp^Read_102/DisableCopyOnRead^Read_102/ReadVariableOp^Read_103/DisableCopyOnRead^Read_103/ReadVariableOp^Read_104/DisableCopyOnRead^Read_104/ReadVariableOp^Read_105/DisableCopyOnRead^Read_105/ReadVariableOp^Read_106/DisableCopyOnRead^Read_106/ReadVariableOp^Read_107/DisableCopyOnRead^Read_107/ReadVariableOp^Read_108/DisableCopyOnRead^Read_108/ReadVariableOp^Read_109/DisableCopyOnRead^Read_109/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_110/DisableCopyOnRead^Read_110/ReadVariableOp^Read_111/DisableCopyOnRead^Read_111/ReadVariableOp^Read_112/DisableCopyOnRead^Read_112/ReadVariableOp^Read_113/DisableCopyOnRead^Read_113/ReadVariableOp^Read_114/DisableCopyOnRead^Read_114/ReadVariableOp^Read_115/DisableCopyOnRead^Read_115/ReadVariableOp^Read_116/DisableCopyOnRead^Read_116/ReadVariableOp^Read_117/DisableCopyOnRead^Read_117/ReadVariableOp^Read_118/DisableCopyOnRead^Read_118/ReadVariableOp^Read_119/DisableCopyOnRead^Read_119/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_120/DisableCopyOnRead^Read_120/ReadVariableOp^Read_121/DisableCopyOnRead^Read_121/ReadVariableOp^Read_122/DisableCopyOnRead^Read_122/ReadVariableOp^Read_123/DisableCopyOnRead^Read_123/ReadVariableOp^Read_124/DisableCopyOnRead^Read_124/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_88/DisableCopyOnRead^Read_88/ReadVariableOp^Read_89/DisableCopyOnRead^Read_89/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp^Read_90/DisableCopyOnRead^Read_90/ReadVariableOp^Read_91/DisableCopyOnRead^Read_91/ReadVariableOp^Read_92/DisableCopyOnRead^Read_92/ReadVariableOp^Read_93/DisableCopyOnRead^Read_93/ReadVariableOp^Read_94/DisableCopyOnRead^Read_94/ReadVariableOp^Read_95/DisableCopyOnRead^Read_95/ReadVariableOp^Read_96/DisableCopyOnRead^Read_96/ReadVariableOp^Read_97/DisableCopyOnRead^Read_97/ReadVariableOp^Read_98/DisableCopyOnRead^Read_98/ReadVariableOp^Read_99/DisableCopyOnRead^Read_99/ReadVariableOp*
_output_shapes
 "%
identity_251Identity_251:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp28
Read_100/DisableCopyOnReadRead_100/DisableCopyOnRead22
Read_100/ReadVariableOpRead_100/ReadVariableOp28
Read_101/DisableCopyOnReadRead_101/DisableCopyOnRead22
Read_101/ReadVariableOpRead_101/ReadVariableOp28
Read_102/DisableCopyOnReadRead_102/DisableCopyOnRead22
Read_102/ReadVariableOpRead_102/ReadVariableOp28
Read_103/DisableCopyOnReadRead_103/DisableCopyOnRead22
Read_103/ReadVariableOpRead_103/ReadVariableOp28
Read_104/DisableCopyOnReadRead_104/DisableCopyOnRead22
Read_104/ReadVariableOpRead_104/ReadVariableOp28
Read_105/DisableCopyOnReadRead_105/DisableCopyOnRead22
Read_105/ReadVariableOpRead_105/ReadVariableOp28
Read_106/DisableCopyOnReadRead_106/DisableCopyOnRead22
Read_106/ReadVariableOpRead_106/ReadVariableOp28
Read_107/DisableCopyOnReadRead_107/DisableCopyOnRead22
Read_107/ReadVariableOpRead_107/ReadVariableOp28
Read_108/DisableCopyOnReadRead_108/DisableCopyOnRead22
Read_108/ReadVariableOpRead_108/ReadVariableOp28
Read_109/DisableCopyOnReadRead_109/DisableCopyOnRead22
Read_109/ReadVariableOpRead_109/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp28
Read_110/DisableCopyOnReadRead_110/DisableCopyOnRead22
Read_110/ReadVariableOpRead_110/ReadVariableOp28
Read_111/DisableCopyOnReadRead_111/DisableCopyOnRead22
Read_111/ReadVariableOpRead_111/ReadVariableOp28
Read_112/DisableCopyOnReadRead_112/DisableCopyOnRead22
Read_112/ReadVariableOpRead_112/ReadVariableOp28
Read_113/DisableCopyOnReadRead_113/DisableCopyOnRead22
Read_113/ReadVariableOpRead_113/ReadVariableOp28
Read_114/DisableCopyOnReadRead_114/DisableCopyOnRead22
Read_114/ReadVariableOpRead_114/ReadVariableOp28
Read_115/DisableCopyOnReadRead_115/DisableCopyOnRead22
Read_115/ReadVariableOpRead_115/ReadVariableOp28
Read_116/DisableCopyOnReadRead_116/DisableCopyOnRead22
Read_116/ReadVariableOpRead_116/ReadVariableOp28
Read_117/DisableCopyOnReadRead_117/DisableCopyOnRead22
Read_117/ReadVariableOpRead_117/ReadVariableOp28
Read_118/DisableCopyOnReadRead_118/DisableCopyOnRead22
Read_118/ReadVariableOpRead_118/ReadVariableOp28
Read_119/DisableCopyOnReadRead_119/DisableCopyOnRead22
Read_119/ReadVariableOpRead_119/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp28
Read_120/DisableCopyOnReadRead_120/DisableCopyOnRead22
Read_120/ReadVariableOpRead_120/ReadVariableOp28
Read_121/DisableCopyOnReadRead_121/DisableCopyOnRead22
Read_121/ReadVariableOpRead_121/ReadVariableOp28
Read_122/DisableCopyOnReadRead_122/DisableCopyOnRead22
Read_122/ReadVariableOpRead_122/ReadVariableOp28
Read_123/DisableCopyOnReadRead_123/DisableCopyOnRead22
Read_123/ReadVariableOpRead_123/ReadVariableOp28
Read_124/DisableCopyOnReadRead_124/DisableCopyOnRead22
Read_124/ReadVariableOpRead_124/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp26
Read_88/DisableCopyOnReadRead_88/DisableCopyOnRead20
Read_88/ReadVariableOpRead_88/ReadVariableOp26
Read_89/DisableCopyOnReadRead_89/DisableCopyOnRead20
Read_89/ReadVariableOpRead_89/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp26
Read_90/DisableCopyOnReadRead_90/DisableCopyOnRead20
Read_90/ReadVariableOpRead_90/ReadVariableOp26
Read_91/DisableCopyOnReadRead_91/DisableCopyOnRead20
Read_91/ReadVariableOpRead_91/ReadVariableOp26
Read_92/DisableCopyOnReadRead_92/DisableCopyOnRead20
Read_92/ReadVariableOpRead_92/ReadVariableOp26
Read_93/DisableCopyOnReadRead_93/DisableCopyOnRead20
Read_93/ReadVariableOpRead_93/ReadVariableOp26
Read_94/DisableCopyOnReadRead_94/DisableCopyOnRead20
Read_94/ReadVariableOpRead_94/ReadVariableOp26
Read_95/DisableCopyOnReadRead_95/DisableCopyOnRead20
Read_95/ReadVariableOpRead_95/ReadVariableOp26
Read_96/DisableCopyOnReadRead_96/DisableCopyOnRead20
Read_96/ReadVariableOpRead_96/ReadVariableOp26
Read_97/DisableCopyOnReadRead_97/DisableCopyOnRead20
Read_97/ReadVariableOpRead_97/ReadVariableOp26
Read_98/DisableCopyOnReadRead_98/DisableCopyOnRead20
Read_98/ReadVariableOpRead_98/ReadVariableOp26
Read_99/DisableCopyOnReadRead_99/DisableCopyOnRead20
Read_99/ReadVariableOpRead_99/ReadVariableOp:=~9

_output_shapes
: 

_user_specified_nameConst:6}2
0
_user_specified_namerandom_zoom_2/StateVar::|6
4
_user_specified_namerandom_rotation_2/StateVar:6{2
0
_user_specified_namerandom_flip_2/StateVar:%z!

_user_specified_namecount:%y!

_user_specified_nametotal:'x#
!
_user_specified_name	count_1:'w#
!
_user_specified_name	total_1:=v9
7
_user_specified_nameRMSprop/velocity/dense_2/bias:?u;
9
_user_specified_name!RMSprop/velocity/dense_2/kernel:At=
;
_user_specified_name#!RMSprop/velocity/conv2d_15/kernel:TsP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_8/pointwise_kernel:TrP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_8/depthwise_kernel:KqG
E
_user_specified_name-+RMSprop/velocity/batch_normalization_8/beta:LpH
F
_user_specified_name.,RMSprop/velocity/batch_normalization_8/gamma:ToP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_9/pointwise_kernel:TnP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_9/depthwise_kernel:KmG
E
_user_specified_name-+RMSprop/velocity/batch_normalization_9/beta:LlH
F
_user_specified_name.,RMSprop/velocity/batch_normalization_9/gamma:Ak=
;
_user_specified_name#!RMSprop/velocity/conv2d_14/kernel:TjP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_6/pointwise_kernel:TiP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_6/depthwise_kernel:KhG
E
_user_specified_name-+RMSprop/velocity/batch_normalization_6/beta:LgH
F
_user_specified_name.,RMSprop/velocity/batch_normalization_6/gamma:TfP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_7/pointwise_kernel:TeP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_7/depthwise_kernel:KdG
E
_user_specified_name-+RMSprop/velocity/batch_normalization_7/beta:LcH
F
_user_specified_name.,RMSprop/velocity/batch_normalization_7/gamma:Ab=
;
_user_specified_name#!RMSprop/velocity/conv2d_13/kernel:TaP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_4/pointwise_kernel:T`P
N
_user_specified_name64RMSprop/velocity/separable_conv2d_4/depthwise_kernel:K_G
E
_user_specified_name-+RMSprop/velocity/batch_normalization_4/beta:L^H
F
_user_specified_name.,RMSprop/velocity/batch_normalization_4/gamma:T]P
N
_user_specified_name64RMSprop/velocity/separable_conv2d_5/pointwise_kernel:T\P
N
_user_specified_name64RMSprop/velocity/separable_conv2d_5/depthwise_kernel:K[G
E
_user_specified_name-+RMSprop/velocity/batch_normalization_5/beta:LZH
F
_user_specified_name.,RMSprop/velocity/batch_normalization_5/gamma:AY=
;
_user_specified_name#!RMSprop/velocity/conv2d_12/kernel:TXP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_2/pointwise_kernel:TWP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_2/depthwise_kernel:KVG
E
_user_specified_name-+RMSprop/velocity/batch_normalization_2/beta:LUH
F
_user_specified_name.,RMSprop/velocity/batch_normalization_2/gamma:TTP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_3/pointwise_kernel:TSP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_3/depthwise_kernel:KRG
E
_user_specified_name-+RMSprop/velocity/batch_normalization_3/beta:LQH
F
_user_specified_name.,RMSprop/velocity/batch_normalization_3/gamma:AP=
;
_user_specified_name#!RMSprop/velocity/conv2d_11/kernel:RON
L
_user_specified_name42RMSprop/velocity/separable_conv2d/pointwise_kernel:RNN
L
_user_specified_name42RMSprop/velocity/separable_conv2d/depthwise_kernel:IME
C
_user_specified_name+)RMSprop/velocity/batch_normalization/beta:JLF
D
_user_specified_name,*RMSprop/velocity/batch_normalization/gamma:TKP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_1/pointwise_kernel:TJP
N
_user_specified_name64RMSprop/velocity/separable_conv2d_1/depthwise_kernel:KIG
E
_user_specified_name-+RMSprop/velocity/batch_normalization_1/beta:LHH
F
_user_specified_name.,RMSprop/velocity/batch_normalization_1/gamma:AG=
;
_user_specified_name#!RMSprop/velocity/conv2d_10/kernel:-F)
'
_user_specified_namelearning_rate:)E%
#
_user_specified_name	iteration:,D(
&
_user_specified_namedense_2/bias:.C*
(
_user_specified_namedense_2/kernel:0B,
*
_user_specified_nameconv2d_15/kernel:CA?
=
_user_specified_name%#separable_conv2d_8/pointwise_kernel:C@?
=
_user_specified_name%#separable_conv2d_8/depthwise_kernel:E?A
?
_user_specified_name'%batch_normalization_8/moving_variance:A>=
;
_user_specified_name#!batch_normalization_8/moving_mean::=6
4
_user_specified_namebatch_normalization_8/beta:;<7
5
_user_specified_namebatch_normalization_8/gamma:C;?
=
_user_specified_name%#separable_conv2d_9/pointwise_kernel:C:?
=
_user_specified_name%#separable_conv2d_9/depthwise_kernel:E9A
?
_user_specified_name'%batch_normalization_9/moving_variance:A8=
;
_user_specified_name#!batch_normalization_9/moving_mean::76
4
_user_specified_namebatch_normalization_9/beta:;67
5
_user_specified_namebatch_normalization_9/gamma:05,
*
_user_specified_nameconv2d_14/kernel:C4?
=
_user_specified_name%#separable_conv2d_6/pointwise_kernel:C3?
=
_user_specified_name%#separable_conv2d_6/depthwise_kernel:E2A
?
_user_specified_name'%batch_normalization_6/moving_variance:A1=
;
_user_specified_name#!batch_normalization_6/moving_mean::06
4
_user_specified_namebatch_normalization_6/beta:;/7
5
_user_specified_namebatch_normalization_6/gamma:C.?
=
_user_specified_name%#separable_conv2d_7/pointwise_kernel:C-?
=
_user_specified_name%#separable_conv2d_7/depthwise_kernel:E,A
?
_user_specified_name'%batch_normalization_7/moving_variance:A+=
;
_user_specified_name#!batch_normalization_7/moving_mean::*6
4
_user_specified_namebatch_normalization_7/beta:;)7
5
_user_specified_namebatch_normalization_7/gamma:0(,
*
_user_specified_nameconv2d_13/kernel:C'?
=
_user_specified_name%#separable_conv2d_4/pointwise_kernel:C&?
=
_user_specified_name%#separable_conv2d_4/depthwise_kernel:E%A
?
_user_specified_name'%batch_normalization_4/moving_variance:A$=
;
_user_specified_name#!batch_normalization_4/moving_mean::#6
4
_user_specified_namebatch_normalization_4/beta:;"7
5
_user_specified_namebatch_normalization_4/gamma:C!?
=
_user_specified_name%#separable_conv2d_5/pointwise_kernel:C ?
=
_user_specified_name%#separable_conv2d_5/depthwise_kernel:EA
?
_user_specified_name'%batch_normalization_5/moving_variance:A=
;
_user_specified_name#!batch_normalization_5/moving_mean::6
4
_user_specified_namebatch_normalization_5/beta:;7
5
_user_specified_namebatch_normalization_5/gamma:0,
*
_user_specified_nameconv2d_12/kernel:C?
=
_user_specified_name%#separable_conv2d_2/pointwise_kernel:C?
=
_user_specified_name%#separable_conv2d_2/depthwise_kernel:EA
?
_user_specified_name'%batch_normalization_2/moving_variance:A=
;
_user_specified_name#!batch_normalization_2/moving_mean::6
4
_user_specified_namebatch_normalization_2/beta:;7
5
_user_specified_namebatch_normalization_2/gamma:C?
=
_user_specified_name%#separable_conv2d_3/pointwise_kernel:C?
=
_user_specified_name%#separable_conv2d_3/depthwise_kernel:EA
?
_user_specified_name'%batch_normalization_3/moving_variance:A=
;
_user_specified_name#!batch_normalization_3/moving_mean::6
4
_user_specified_namebatch_normalization_3/beta:;7
5
_user_specified_namebatch_normalization_3/gamma:0,
*
_user_specified_nameconv2d_11/kernel:A=
;
_user_specified_name#!separable_conv2d/pointwise_kernel:A=
;
_user_specified_name#!separable_conv2d/depthwise_kernel:C?
=
_user_specified_name%#batch_normalization/moving_variance:?
;
9
_user_specified_name!batch_normalization/moving_mean:8	4
2
_user_specified_namebatch_normalization/beta:95
3
_user_specified_namebatch_normalization/gamma:C?
=
_user_specified_name%#separable_conv2d_1/pointwise_kernel:C?
=
_user_specified_name%#separable_conv2d_1/depthwise_kernel:EA
?
_user_specified_name'%batch_normalization_1/moving_variance:A=
;
_user_specified_name#!batch_normalization_1/moving_mean::6
4
_user_specified_namebatch_normalization_1/beta:;7
5
_user_specified_namebatch_normalization_1/gamma:0,
*
_user_specified_nameconv2d_10/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_184214

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_10_layer_call_and_return_conditional_losses_182285

inputs8
conv2d_readvariableop_resource: 
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingVALID*
strides
h
IdentityIdentityConv2D:output:0^NoOp*
T0*1
_output_shapes
:����������� :
NoOpNoOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
3__inference_separable_conv2d_3_layer_call_fn_183714

inputs!
unknown: #
	unknown_0: @
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_181619�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183710:&"
 
_user_specified_name183708:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
R
&__inference_add_1_layer_call_fn_183849
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������??@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_182403h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������??@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������??@:���������??@:YU
/
_output_shapes
:���������??@
"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������??@
"
_user_specified_name
inputs_0
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_181581

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
d
H__inference_activation_3_layer_call_and_return_conditional_losses_183705

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������~~ b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������~~ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������~~ :W S
/
_output_shapes
:���������~~ 
 
_user_specified_nameinputs
�
I
-__inference_activation_9_layer_call_fn_184366

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_9_layer_call_and_return_conditional_losses_182534i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_183473

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
N__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_182054

inputsC
(separable_conv2d_readvariableop_resource:�F
*separable_conv2d_readvariableop_1_resource:��
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:��*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
e
I__inference_random_zoom_2_layer_call_and_return_conditional_losses_181314

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_15_layer_call_fn_184502

inputs#
unknown:��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_182568x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:����������: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184498:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
b
F__inference_activation_layer_call_and_return_conditional_losses_182322

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:����������� d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:����������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:����������� :Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�
�
N__inference_separable_conv2d_9_layer_call_and_return_conditional_losses_184392

inputsC
(separable_conv2d_readvariableop_resource:�F
*separable_conv2d_readvariableop_1_resource:��
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:��*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_182091

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_182016

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_12_layer_call_fn_184490

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_182244�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�	
�
3__inference_separable_conv2d_6_layer_call_fn_184251

inputs"
unknown:�%
	unknown_0:��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_182054�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184247:&"
 
_user_specified_name184245:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
k
A__inference_add_4_layer_call_and_return_conditional_losses_182577

inputs
inputs_1
identityY
addAddV2inputsinputs_1*
T0*0
_output_shapes
:����������X
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������:����������:XT
0
_output_shapes
:����������
 
_user_specified_nameinputs:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
m
A__inference_add_2_layer_call_and_return_conditional_losses_184077
inputs_0
inputs_1
identity[
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:���������  �X
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������  �:���������  �:ZV
0
_output_shapes
:���������  �
"
_user_specified_name
inputs_1:Z V
0
_output_shapes
:���������  �
"
_user_specified_name
inputs_0
�

�
4__inference_batch_normalization_layer_call_fn_183517

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_181470�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183513:&"
 
_user_specified_name183511:&"
 
_user_specified_name183509:&"
 
_user_specified_name183507:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
E__inference_conv2d_11_layer_call_and_return_conditional_losses_182336

inputs8
conv2d_readvariableop_resource:  
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������~~ *
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:���������~~ :
NoOpNoOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:����������� : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�
�
*__inference_conv2d_14_layer_call_fn_184280

inputs#
unknown:��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_182510x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:���������  �: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184276:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
d
H__inference_activation_2_layer_call_and_return_conditional_losses_182380

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������~~@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������~~@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������~~@:W S
/
_output_shapes
:���������~~@
 
_user_specified_nameinputs
�
i
M__inference_random_rotation_2_layer_call_and_return_conditional_losses_181309

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_183455

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_182174

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_181664

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_181646

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�M
�
I__inference_random_flip_2_layer_call_and_return_conditional_losses_184649

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity��(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :�
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask�
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:�
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :�
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask�
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:������������
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
::��~
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:�
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    �
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::�
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0_stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:����������
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: �
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:����������
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:���������r
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:���������y
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:�
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:������������
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:�����������k
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:����������
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:������������
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:������������
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*1
_output_shapes
:�����������M
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:($
"
_user_specified_name
resource:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_12_layer_call_fn_183836

inputs!
unknown: @
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������??@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_182394w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������??@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������~~ : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183832:W S
/
_output_shapes
:���������~~ 
 
_user_specified_nameinputs
�
H
,__inference_rescaling_2_layer_call_fn_183389

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_rescaling_2_layer_call_and_return_conditional_losses_182277j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
d
H__inference_activation_8_layer_call_and_return_conditional_losses_182554

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:����������c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_11_layer_call_fn_184268

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_182068�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
J
.__inference_random_flip_2_layer_call_fn_184591

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_random_flip_2_layer_call_and_return_conditional_losses_181304j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
m
A__inference_add_4_layer_call_and_return_conditional_losses_184521
inputs_0
inputs_1
identity[
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:����������X
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������:����������:ZV
0
_output_shapes
:����������
"
_user_specified_name
inputs_1:Z V
0
_output_shapes
:����������
"
_user_specified_name
inputs_0
�
c
*__inference_dropout_1_layer_call_fn_184537

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_182591p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_183819

inputsB
(separable_conv2d_readvariableop_resource:@D
*separable_conv2d_readvariableop_1_resource:@@
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:@@*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
E__inference_conv2d_10_layer_call_and_return_conditional_losses_183411

inputs8
conv2d_readvariableop_resource: 
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingVALID*
strides
h
IdentityIdentityConv2D:output:0^NoOp*
T0*1
_output_shapes
:����������� :
NoOpNoOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
e
I__inference_random_flip_2_layer_call_and_return_conditional_losses_184653

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_184273

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_13_layer_call_and_return_conditional_losses_184065

inputs9
conv2d_readvariableop_resource:@�
identity��Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������  �*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:���������  �:
NoOpNoOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������??@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������??@
 
_user_specified_nameinputs
�
L
0__inference_max_pooling2d_8_layer_call_fn_183602

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_181540�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_1_layer_call_fn_183424

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_181387�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183420:&"
 
_user_specified_name183418:&"
 
_user_specified_name183416:&"
 
_user_specified_name183414:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
b
F__inference_activation_layer_call_and_return_conditional_losses_183576

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:����������� d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:����������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:����������� :Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�
d
H__inference_activation_6_layer_call_and_return_conditional_losses_182496

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:���������  �c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������  �:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�

�
4__inference_batch_normalization_layer_call_fn_183530

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_181488�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183526:&"
 
_user_specified_name183524:&"
 
_user_specified_name183522:&"
 
_user_specified_name183520:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
d
H__inference_activation_6_layer_call_and_return_conditional_losses_184242

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:���������  �c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������  �:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
d
H__inference_activation_5_layer_call_and_return_conditional_losses_182418

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������??@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������??@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������??@:W S
/
_output_shapes
:���������??@
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_183695

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_7_layer_call_fn_184090

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_181915�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184086:&"
 
_user_specified_name184084:&"
 
_user_specified_name184082:&"
 
_user_specified_name184080:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
I
-__inference_activation_4_layer_call_fn_184015

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������??�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_182438i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������??�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������??�:X T
0
_output_shapes
:���������??�
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_5_layer_call_fn_183881

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_181757�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183877:&"
 
_user_specified_name183875:&"
 
_user_specified_name183873:&"
 
_user_specified_name183871:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_181892

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
k
A__inference_add_1_layer_call_and_return_conditional_losses_182403

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������??@W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������??@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������??@:���������??@:WS
/
_output_shapes
:���������??@
 
_user_specified_nameinputs:W S
/
_output_shapes
:���������??@
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_4_layer_call_fn_183974

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_181840�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183970:&"
 
_user_specified_name183968:&"
 
_user_specified_name183966:&"
 
_user_specified_name183964:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
d
H__inference_activation_4_layer_call_and_return_conditional_losses_184020

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:���������??�c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:���������??�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������??�:X T
0
_output_shapes
:���������??�
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_181563

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_6_layer_call_fn_184196

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_182016�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184192:&"
 
_user_specified_name184190:&"
 
_user_specified_name184188:&"
 
_user_specified_name184186:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
k
A__inference_add_2_layer_call_and_return_conditional_losses_182461

inputs
inputs_1
identityY
addAddV2inputsinputs_1*
T0*0
_output_shapes
:���������  �X
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������  �:���������  �:XT
0
_output_shapes
:���������  �
 
_user_specified_nameinputs:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_10_layer_call_fn_184046

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_181892�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
c
G__inference_rescaling_2_layer_call_and_return_conditional_losses_183397

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *���;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:�����������d
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:�����������Y
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_181915

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
k
?__inference_add_layer_call_and_return_conditional_losses_183633
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������~~ W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������~~ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������~~ :���������~~ :YU
/
_output_shapes
:���������~~ 
"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������~~ 
"
_user_specified_name
inputs_0
�

�
6__inference_batch_normalization_4_layer_call_fn_183961

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_181822�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183957:&"
 
_user_specified_name183955:&"
 
_user_specified_name183953:&"
 
_user_specified_name183951:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
i
M__inference_random_rotation_2_layer_call_and_return_conditional_losses_184787

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
p
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_184532

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_181619

inputsB
(separable_conv2d_readvariableop_resource: D
*separable_conv2d_readvariableop_1_resource: @
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
: @*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
U
9__inference_global_average_pooling2d_layer_call_fn_184526

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_182255i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_6_layer_call_fn_184183

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_181998�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184179:&"
 
_user_specified_name184177:&"
 
_user_specified_name184175:&"
 
_user_specified_name184173:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
L__inference_separable_conv2d_layer_call_and_return_conditional_losses_183597

inputsB
(separable_conv2d_readvariableop_resource: D
*separable_conv2d_readvariableop_1_resource:  
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:  *
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_184232

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_184436

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_183829

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_11_layer_call_fn_183614

inputs!
unknown:  
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_182336w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������~~ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:����������� : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183610:Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�
�
L__inference_separable_conv2d_layer_call_and_return_conditional_losses_181526

inputsB
(separable_conv2d_readvariableop_resource: D
*separable_conv2d_readvariableop_1_resource:  
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:  *
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_7_layer_call_fn_184103

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_181933�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184099:&"
 
_user_specified_name184097:&"
 
_user_specified_name184095:&"
 
_user_specified_name184093:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_10_layer_call_fn_183404

inputs!
unknown: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_182285y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:����������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183400:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_13_layer_call_fn_184058

inputs"
unknown:@�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_182452x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������  �<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������??@: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184054:W S
/
_output_shapes
:���������??@
 
_user_specified_nameinputs
�
J
.__inference_random_zoom_2_layer_call_fn_184799

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_random_zoom_2_layer_call_and_return_conditional_losses_181314j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_183726

inputsB
(separable_conv2d_readvariableop_resource: D
*separable_conv2d_readvariableop_1_resource: @
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
: @*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_3_layer_call_fn_183646

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_181563�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183642:&"
 
_user_specified_name183640:&"
 
_user_specified_name183638:&"
 
_user_specified_name183636:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
d
H__inference_activation_1_layer_call_and_return_conditional_losses_183483

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:����������� d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:����������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:����������� :Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�
k
A__inference_add_3_layer_call_and_return_conditional_losses_182519

inputs
inputs_1
identityY
addAddV2inputsinputs_1*
T0*0
_output_shapes
:����������X
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������:����������:XT
0
_output_shapes
:����������
 
_user_specified_nameinputs:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_181405

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
i
?__inference_add_layer_call_and_return_conditional_losses_182345

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������~~ W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������~~ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������~~ :���������~~ :WS
/
_output_shapes
:���������~~ 
 
_user_specified_nameinputs:W S
/
_output_shapes
:���������~~ 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_184343

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�M
�
I__inference_random_flip_2_layer_call_and_return_conditional_losses_181067

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity��(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :�
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask�
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:�
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :�
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask�
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:������������
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
::��~
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:�
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    �
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::�
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0_stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:����������
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: �
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:����������
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:���������r
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :�
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:�
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:����������
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:���������y
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:�
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:������������
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:�����������k
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:����������
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:������������
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:������������
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*1
_output_shapes
:�����������M
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:($
"
_user_specified_name
resource:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_182798

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
ޅ
�$
C__inference_model_2_layer_call_and_return_conditional_losses_182610
input_3!
sequential_2_182263:	!
sequential_2_182265:	!
sequential_2_182267:	*
conv2d_10_182286: *
batch_normalization_1_182289: *
batch_normalization_1_182291: *
batch_normalization_1_182293: *
batch_normalization_1_182295: 3
separable_conv2d_1_182304: 3
separable_conv2d_1_182306:  (
batch_normalization_182309: (
batch_normalization_182311: (
batch_normalization_182313: (
batch_normalization_182315: 1
separable_conv2d_182324: 1
separable_conv2d_182326:  *
conv2d_11_182337:  *
batch_normalization_3_182347: *
batch_normalization_3_182349: *
batch_normalization_3_182351: *
batch_normalization_3_182353: 3
separable_conv2d_3_182362: 3
separable_conv2d_3_182364: @*
batch_normalization_2_182367:@*
batch_normalization_2_182369:@*
batch_normalization_2_182371:@*
batch_normalization_2_182373:@3
separable_conv2d_2_182382:@3
separable_conv2d_2_182384:@@*
conv2d_12_182395: @*
batch_normalization_5_182405:@*
batch_normalization_5_182407:@*
batch_normalization_5_182409:@*
batch_normalization_5_182411:@3
separable_conv2d_5_182420:@4
separable_conv2d_5_182422:@�+
batch_normalization_4_182425:	�+
batch_normalization_4_182427:	�+
batch_normalization_4_182429:	�+
batch_normalization_4_182431:	�4
separable_conv2d_4_182440:�5
separable_conv2d_4_182442:��+
conv2d_13_182453:@�+
batch_normalization_7_182463:	�+
batch_normalization_7_182465:	�+
batch_normalization_7_182467:	�+
batch_normalization_7_182469:	�4
separable_conv2d_7_182478:�5
separable_conv2d_7_182480:��+
batch_normalization_6_182483:	�+
batch_normalization_6_182485:	�+
batch_normalization_6_182487:	�+
batch_normalization_6_182489:	�4
separable_conv2d_6_182498:�5
separable_conv2d_6_182500:��,
conv2d_14_182511:��+
batch_normalization_9_182521:	�+
batch_normalization_9_182523:	�+
batch_normalization_9_182525:	�+
batch_normalization_9_182527:	�4
separable_conv2d_9_182536:�5
separable_conv2d_9_182538:��+
batch_normalization_8_182541:	�+
batch_normalization_8_182543:	�+
batch_normalization_8_182545:	�+
batch_normalization_8_182547:	�4
separable_conv2d_8_182556:�5
separable_conv2d_8_182558:��,
conv2d_15_182569:��!
dense_2_182604:	�
dense_2_182606:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�-batch_normalization_8/StatefulPartitionedCall�-batch_normalization_9/StatefulPartitionedCall�!conv2d_10/StatefulPartitionedCall�!conv2d_11/StatefulPartitionedCall�!conv2d_12/StatefulPartitionedCall�!conv2d_13/StatefulPartitionedCall�!conv2d_14/StatefulPartitionedCall�!conv2d_15/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�(separable_conv2d/StatefulPartitionedCall�*separable_conv2d_1/StatefulPartitionedCall�*separable_conv2d_2/StatefulPartitionedCall�*separable_conv2d_3/StatefulPartitionedCall�*separable_conv2d_4/StatefulPartitionedCall�*separable_conv2d_5/StatefulPartitionedCall�*separable_conv2d_6/StatefulPartitionedCall�*separable_conv2d_7/StatefulPartitionedCall�*separable_conv2d_8/StatefulPartitionedCall�*separable_conv2d_9/StatefulPartitionedCall�$sequential_2/StatefulPartitionedCall�
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinput_3sequential_2_182263sequential_2_182265sequential_2_182267*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_181298�
rescaling_2/PartitionedCallPartitionedCall-sequential_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_rescaling_2_layer_call_and_return_conditional_losses_182277�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall$rescaling_2/PartitionedCall:output:0conv2d_10_182286*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_182285�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0batch_normalization_1_182289batch_normalization_1_182291batch_normalization_1_182293batch_normalization_1_182295*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_181387�
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_182302�
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0separable_conv2d_1_182304separable_conv2d_1_182306*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_181443�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0batch_normalization_182309batch_normalization_182311batch_normalization_182313batch_normalization_182315*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_181470�
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_182322�
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0separable_conv2d_182324separable_conv2d_182326*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_separable_conv2d_layer_call_and_return_conditional_losses_181526�
max_pooling2d_8/PartitionedCallPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_181540�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_182337*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~ *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_182336�
add/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_182345�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0batch_normalization_3_182347batch_normalization_3_182349batch_normalization_3_182351batch_normalization_3_182353*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_181563�
activation_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_182360�
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0separable_conv2d_3_182362separable_conv2d_3_182364*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_181619�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0batch_normalization_2_182367batch_normalization_2_182369batch_normalization_2_182371batch_normalization_2_182373*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_181646�
activation_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_182380�
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0separable_conv2d_2_182382separable_conv2d_2_182384*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_181702�
max_pooling2d_9/PartitionedCallPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������??@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_181716�
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0conv2d_12_182395*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������??@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_182394�
add_1/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������??@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_182403�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0batch_normalization_5_182405batch_normalization_5_182407batch_normalization_5_182409batch_normalization_5_182411*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������??@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_181739�
activation_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������??@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_182418�
*separable_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0separable_conv2d_5_182420separable_conv2d_5_182422*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������??�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_181795�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_5/StatefulPartitionedCall:output:0batch_normalization_4_182425batch_normalization_4_182427batch_normalization_4_182429batch_normalization_4_182431*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������??�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_181822�
activation_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������??�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_182438�
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0separable_conv2d_4_182440separable_conv2d_4_182442*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������??�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_181878�
 max_pooling2d_10/PartitionedCallPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_181892�
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0conv2d_13_182453*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_182452�
add_2/PartitionedCallPartitionedCall)max_pooling2d_10/PartitionedCall:output:0*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_182461�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0batch_normalization_7_182463batch_normalization_7_182465batch_normalization_7_182467batch_normalization_7_182469*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_181915�
activation_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_7_layer_call_and_return_conditional_losses_182476�
*separable_conv2d_7/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:0separable_conv2d_7_182478separable_conv2d_7_182480*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_181971�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_7/StatefulPartitionedCall:output:0batch_normalization_6_182483batch_normalization_6_182485batch_normalization_6_182487batch_normalization_6_182489*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_181998�
activation_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_182496�
*separable_conv2d_6/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0separable_conv2d_6_182498separable_conv2d_6_182500*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������  �*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_182054�
 max_pooling2d_11/PartitionedCallPartitionedCall3separable_conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_182068�
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0conv2d_14_182511*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_182510�
add_3/PartitionedCallPartitionedCall)max_pooling2d_11/PartitionedCall:output:0*conv2d_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_182519�
-batch_normalization_9/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0batch_normalization_9_182521batch_normalization_9_182523batch_normalization_9_182525batch_normalization_9_182527*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_182091�
activation_9/PartitionedCallPartitionedCall6batch_normalization_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_9_layer_call_and_return_conditional_losses_182534�
*separable_conv2d_9/StatefulPartitionedCallStatefulPartitionedCall%activation_9/PartitionedCall:output:0separable_conv2d_9_182536separable_conv2d_9_182538*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_9_layer_call_and_return_conditional_losses_182147�
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_9/StatefulPartitionedCall:output:0batch_normalization_8_182541batch_normalization_8_182543batch_normalization_8_182545batch_normalization_8_182547*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_182174�
activation_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_8_layer_call_and_return_conditional_losses_182554�
*separable_conv2d_8/StatefulPartitionedCallStatefulPartitionedCall%activation_8/PartitionedCall:output:0separable_conv2d_8_182556separable_conv2d_8_182558*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_8_layer_call_and_return_conditional_losses_182230�
 max_pooling2d_12/PartitionedCallPartitionedCall3separable_conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_182244�
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0conv2d_15_182569*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_182568�
add_4/PartitionedCallPartitionedCall)max_pooling2d_12/PartitionedCall:output:0*conv2d_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_4_layer_call_and_return_conditional_losses_182577�
(global_average_pooling2d/PartitionedCallPartitionedCalladd_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_182255�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_182591�
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_182604dense_2_182606*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_182603w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������

NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall.^batch_normalization_9/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall+^separable_conv2d_5/StatefulPartitionedCall+^separable_conv2d_6/StatefulPartitionedCall+^separable_conv2d_7/StatefulPartitionedCall+^separable_conv2d_8/StatefulPartitionedCall+^separable_conv2d_9/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2^
-batch_normalization_9/StatefulPartitionedCall-batch_normalization_9/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall2X
*separable_conv2d_5/StatefulPartitionedCall*separable_conv2d_5/StatefulPartitionedCall2X
*separable_conv2d_6/StatefulPartitionedCall*separable_conv2d_6/StatefulPartitionedCall2X
*separable_conv2d_7/StatefulPartitionedCall*separable_conv2d_7/StatefulPartitionedCall2X
*separable_conv2d_8/StatefulPartitionedCall*separable_conv2d_8/StatefulPartitionedCall2X
*separable_conv2d_9/StatefulPartitionedCall*separable_conv2d_9/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:&G"
 
_user_specified_name182606:&F"
 
_user_specified_name182604:&E"
 
_user_specified_name182569:&D"
 
_user_specified_name182558:&C"
 
_user_specified_name182556:&B"
 
_user_specified_name182547:&A"
 
_user_specified_name182545:&@"
 
_user_specified_name182543:&?"
 
_user_specified_name182541:&>"
 
_user_specified_name182538:&="
 
_user_specified_name182536:&<"
 
_user_specified_name182527:&;"
 
_user_specified_name182525:&:"
 
_user_specified_name182523:&9"
 
_user_specified_name182521:&8"
 
_user_specified_name182511:&7"
 
_user_specified_name182500:&6"
 
_user_specified_name182498:&5"
 
_user_specified_name182489:&4"
 
_user_specified_name182487:&3"
 
_user_specified_name182485:&2"
 
_user_specified_name182483:&1"
 
_user_specified_name182480:&0"
 
_user_specified_name182478:&/"
 
_user_specified_name182469:&."
 
_user_specified_name182467:&-"
 
_user_specified_name182465:&,"
 
_user_specified_name182463:&+"
 
_user_specified_name182453:&*"
 
_user_specified_name182442:&)"
 
_user_specified_name182440:&("
 
_user_specified_name182431:&'"
 
_user_specified_name182429:&&"
 
_user_specified_name182427:&%"
 
_user_specified_name182425:&$"
 
_user_specified_name182422:&#"
 
_user_specified_name182420:&""
 
_user_specified_name182411:&!"
 
_user_specified_name182409:& "
 
_user_specified_name182407:&"
 
_user_specified_name182405:&"
 
_user_specified_name182395:&"
 
_user_specified_name182384:&"
 
_user_specified_name182382:&"
 
_user_specified_name182373:&"
 
_user_specified_name182371:&"
 
_user_specified_name182369:&"
 
_user_specified_name182367:&"
 
_user_specified_name182364:&"
 
_user_specified_name182362:&"
 
_user_specified_name182353:&"
 
_user_specified_name182351:&"
 
_user_specified_name182349:&"
 
_user_specified_name182347:&"
 
_user_specified_name182337:&"
 
_user_specified_name182326:&"
 
_user_specified_name182324:&"
 
_user_specified_name182315:&"
 
_user_specified_name182313:&"
 
_user_specified_name182311:&"
 
_user_specified_name182309:&
"
 
_user_specified_name182306:&	"
 
_user_specified_name182304:&"
 
_user_specified_name182295:&"
 
_user_specified_name182293:&"
 
_user_specified_name182291:&"
 
_user_specified_name182289:&"
 
_user_specified_name182286:&"
 
_user_specified_name182267:&"
 
_user_specified_name182265:&"
 
_user_specified_name182263:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_3
�
�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_183917

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
I
-__inference_activation_1_layer_call_fn_183478

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_182302j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:����������� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:����������� :Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_8_layer_call_fn_184418

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_182192�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184414:&"
 
_user_specified_name184412:&"
 
_user_specified_name184410:&"
 
_user_specified_name184408:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
Ȁ
�
M__inference_random_rotation_2_layer_call_and_return_conditional_losses_181188

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity��stateful_uniform/RngReadAndSkipI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|� �Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|� ?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:�
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:���������z
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: �
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:����������
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:���������Z
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?g
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ^
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:���������\
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: |
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:���������^
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:���������\
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ~
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:���������~
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:���������~
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:���������^
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:���������\
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: `
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:���������\
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?k
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: �
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:���������`
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:���������\
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: �
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:����������
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:���������~
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:���������`
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:���������g
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
::��m
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:���������v
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:���������v
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_maskv
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:���������v
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:���������v
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:���������v
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_maskv
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:���������]
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������S
transform/ShapeShapeinputs*
T0*
_output_shapes
::��g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:�����������*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR�
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:�����������D
NoOpNoOp ^stateful_uniform/RngReadAndSkip*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:($
"
_user_specified_name
resource:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_184554

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_181998

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
d
H__inference_activation_3_layer_call_and_return_conditional_losses_182360

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������~~ b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������~~ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������~~ :W S
/
_output_shapes
:���������~~ 
 
_user_specified_nameinputs
�	
�
3__inference_separable_conv2d_5_layer_call_fn_183936

inputs!
unknown:@$
	unknown_0:@�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_181795�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183932:&"
 
_user_specified_name183930:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_9_layer_call_fn_184312

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_182091�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184308:&"
 
_user_specified_name184306:&"
 
_user_specified_name184304:&"
 
_user_specified_name184302:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_5_layer_call_fn_183868

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_181739�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183864:&"
 
_user_specified_name183862:&"
 
_user_specified_name183860:&"
 
_user_specified_name183858:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
N__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_181443

inputsB
(separable_conv2d_readvariableop_resource: D
*separable_conv2d_readvariableop_1_resource:  
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:  *
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
N__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_184170

inputsC
(separable_conv2d_readvariableop_resource:�F
*separable_conv2d_readvariableop_1_resource:��
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:��*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      �      o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_1_layer_call_fn_183437

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_181405�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183433:&"
 
_user_specified_name183431:&"
 
_user_specified_name183429:&"
 
_user_specified_name183427:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�6
�
(__inference_model_2_layer_call_fn_182953
input_3
unknown:	
	unknown_0:	
	unknown_1:	#
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5: 
	unknown_6: #
	unknown_7: #
	unknown_8:  
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: $

unknown_13: $

unknown_14:  $

unknown_15:  

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19: $

unknown_20: $

unknown_21: @

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@$

unknown_26:@$

unknown_27:@@$

unknown_28: @

unknown_29:@

unknown_30:@

unknown_31:@

unknown_32:@$

unknown_33:@%

unknown_34:@�

unknown_35:	�

unknown_36:	�

unknown_37:	�

unknown_38:	�%

unknown_39:�&

unknown_40:��%

unknown_41:@�

unknown_42:	�

unknown_43:	�

unknown_44:	�

unknown_45:	�%

unknown_46:�&

unknown_47:��

unknown_48:	�

unknown_49:	�

unknown_50:	�

unknown_51:	�%

unknown_52:�&

unknown_53:��&

unknown_54:��

unknown_55:	�

unknown_56:	�

unknown_57:	�

unknown_58:	�%

unknown_59:�&

unknown_60:��

unknown_61:	�

unknown_62:	�

unknown_63:	�

unknown_64:	�%

unknown_65:�&

unknown_66:��&

unknown_67:��

unknown_68:	�

unknown_69:
identity��StatefulPartitionedCall�	
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66
unknown_67
unknown_68
unknown_69*S
TinL
J2H*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*R
_read_only_resource_inputs4
20	
 #$%&)*+,-01236789:=>?@CDEFG*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_182610o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&G"
 
_user_specified_name182949:&F"
 
_user_specified_name182947:&E"
 
_user_specified_name182945:&D"
 
_user_specified_name182943:&C"
 
_user_specified_name182941:&B"
 
_user_specified_name182939:&A"
 
_user_specified_name182937:&@"
 
_user_specified_name182935:&?"
 
_user_specified_name182933:&>"
 
_user_specified_name182931:&="
 
_user_specified_name182929:&<"
 
_user_specified_name182927:&;"
 
_user_specified_name182925:&:"
 
_user_specified_name182923:&9"
 
_user_specified_name182921:&8"
 
_user_specified_name182919:&7"
 
_user_specified_name182917:&6"
 
_user_specified_name182915:&5"
 
_user_specified_name182913:&4"
 
_user_specified_name182911:&3"
 
_user_specified_name182909:&2"
 
_user_specified_name182907:&1"
 
_user_specified_name182905:&0"
 
_user_specified_name182903:&/"
 
_user_specified_name182901:&."
 
_user_specified_name182899:&-"
 
_user_specified_name182897:&,"
 
_user_specified_name182895:&+"
 
_user_specified_name182893:&*"
 
_user_specified_name182891:&)"
 
_user_specified_name182889:&("
 
_user_specified_name182887:&'"
 
_user_specified_name182885:&&"
 
_user_specified_name182883:&%"
 
_user_specified_name182881:&$"
 
_user_specified_name182879:&#"
 
_user_specified_name182877:&""
 
_user_specified_name182875:&!"
 
_user_specified_name182873:& "
 
_user_specified_name182871:&"
 
_user_specified_name182869:&"
 
_user_specified_name182867:&"
 
_user_specified_name182865:&"
 
_user_specified_name182863:&"
 
_user_specified_name182861:&"
 
_user_specified_name182859:&"
 
_user_specified_name182857:&"
 
_user_specified_name182855:&"
 
_user_specified_name182853:&"
 
_user_specified_name182851:&"
 
_user_specified_name182849:&"
 
_user_specified_name182847:&"
 
_user_specified_name182845:&"
 
_user_specified_name182843:&"
 
_user_specified_name182841:&"
 
_user_specified_name182839:&"
 
_user_specified_name182837:&"
 
_user_specified_name182835:&"
 
_user_specified_name182833:&"
 
_user_specified_name182831:&"
 
_user_specified_name182829:&
"
 
_user_specified_name182827:&	"
 
_user_specified_name182825:&"
 
_user_specified_name182823:&"
 
_user_specified_name182821:&"
 
_user_specified_name182819:&"
 
_user_specified_name182817:&"
 
_user_specified_name182815:&"
 
_user_specified_name182813:&"
 
_user_specified_name182811:&"
 
_user_specified_name182809:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_3
�
�
N__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_183948

inputsB
(separable_conv2d_readvariableop_resource:@E
*separable_conv2d_readvariableop_1_resource:@�
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*'
_output_shapes
:@�*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
I
-__inference_activation_3_layer_call_fn_183700

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������~~ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_182360h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������~~ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������~~ :W S
/
_output_shapes
:���������~~ 
 
_user_specified_nameinputs
�
m
A__inference_add_1_layer_call_and_return_conditional_losses_183855
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������??@W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������??@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:���������??@:���������??@:YU
/
_output_shapes
:���������??@
"
_user_specified_name
inputs_1:Y U
/
_output_shapes
:���������??@
"
_user_specified_name
inputs_0
�
�
E__inference_conv2d_14_layer_call_and_return_conditional_losses_182510

inputs:
conv2d_readvariableop_resource:��
identity��Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:����������:
NoOpNoOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:���������  �: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�
�
N__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_181971

inputsC
(separable_conv2d_readvariableop_resource:�F
*separable_conv2d_readvariableop_1_resource:��
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:��*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      �      o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_184041

inputsC
(separable_conv2d_readvariableop_resource:�F
*separable_conv2d_readvariableop_1_resource:��
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*'
_output_shapes
:�*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*(
_output_shapes
:��*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      �      o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*B
_output_shapes0
.:,����������������������������*
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�	
�
-__inference_sequential_2_layer_call_fn_181328
random_flip_2_input
unknown:	
	unknown_0:	
	unknown_1:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_2_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_181298y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:�����������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name181324:&"
 
_user_specified_name181322:&"
 
_user_specified_name181320:f b
1
_output_shapes
:�����������
-
_user_specified_namerandom_flip_2_input
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_181488

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
~
.__inference_random_flip_2_layer_call_fn_184586

inputs
unknown:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_random_flip_2_layer_call_and_return_conditional_losses_181067y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184582:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
3__inference_separable_conv2d_1_layer_call_fn_183492

inputs!
unknown: #
	unknown_0:  
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_181443�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+��������������������������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name183488:&"
 
_user_specified_name183486:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�4
�
$__inference_signature_wrapper_183384
input_3!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: #
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: $

unknown_10: $

unknown_11:  $

unknown_12:  

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: $

unknown_18: @

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@$

unknown_23:@$

unknown_24:@@$

unknown_25: @

unknown_26:@

unknown_27:@

unknown_28:@

unknown_29:@$

unknown_30:@%

unknown_31:@�

unknown_32:	�

unknown_33:	�

unknown_34:	�

unknown_35:	�%

unknown_36:�&

unknown_37:��%

unknown_38:@�

unknown_39:	�

unknown_40:	�

unknown_41:	�

unknown_42:	�%

unknown_43:�&

unknown_44:��

unknown_45:	�

unknown_46:	�

unknown_47:	�

unknown_48:	�%

unknown_49:�&

unknown_50:��&

unknown_51:��

unknown_52:	�

unknown_53:	�

unknown_54:	�

unknown_55:	�%

unknown_56:�&

unknown_57:��

unknown_58:	�

unknown_59:	�

unknown_60:	�

unknown_61:	�%

unknown_62:�&

unknown_63:��&

unknown_64:��

unknown_65:	�

unknown_66:
identity��StatefulPartitionedCall�	
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_181007o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&D"
 
_user_specified_name183380:&C"
 
_user_specified_name183378:&B"
 
_user_specified_name183376:&A"
 
_user_specified_name183374:&@"
 
_user_specified_name183372:&?"
 
_user_specified_name183370:&>"
 
_user_specified_name183368:&="
 
_user_specified_name183366:&<"
 
_user_specified_name183364:&;"
 
_user_specified_name183362:&:"
 
_user_specified_name183360:&9"
 
_user_specified_name183358:&8"
 
_user_specified_name183356:&7"
 
_user_specified_name183354:&6"
 
_user_specified_name183352:&5"
 
_user_specified_name183350:&4"
 
_user_specified_name183348:&3"
 
_user_specified_name183346:&2"
 
_user_specified_name183344:&1"
 
_user_specified_name183342:&0"
 
_user_specified_name183340:&/"
 
_user_specified_name183338:&."
 
_user_specified_name183336:&-"
 
_user_specified_name183334:&,"
 
_user_specified_name183332:&+"
 
_user_specified_name183330:&*"
 
_user_specified_name183328:&)"
 
_user_specified_name183326:&("
 
_user_specified_name183324:&'"
 
_user_specified_name183322:&&"
 
_user_specified_name183320:&%"
 
_user_specified_name183318:&$"
 
_user_specified_name183316:&#"
 
_user_specified_name183314:&""
 
_user_specified_name183312:&!"
 
_user_specified_name183310:& "
 
_user_specified_name183308:&"
 
_user_specified_name183306:&"
 
_user_specified_name183304:&"
 
_user_specified_name183302:&"
 
_user_specified_name183300:&"
 
_user_specified_name183298:&"
 
_user_specified_name183296:&"
 
_user_specified_name183294:&"
 
_user_specified_name183292:&"
 
_user_specified_name183290:&"
 
_user_specified_name183288:&"
 
_user_specified_name183286:&"
 
_user_specified_name183284:&"
 
_user_specified_name183282:&"
 
_user_specified_name183280:&"
 
_user_specified_name183278:&"
 
_user_specified_name183276:&"
 
_user_specified_name183274:&"
 
_user_specified_name183272:&"
 
_user_specified_name183270:&"
 
_user_specified_name183268:&"
 
_user_specified_name183266:&
"
 
_user_specified_name183264:&	"
 
_user_specified_name183262:&"
 
_user_specified_name183260:&"
 
_user_specified_name183258:&"
 
_user_specified_name183256:&"
 
_user_specified_name183254:&"
 
_user_specified_name183252:&"
 
_user_specified_name183250:&"
 
_user_specified_name183248:&"
 
_user_specified_name183246:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_3
�
F
*__inference_dropout_1_layer_call_fn_184542

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_182798a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
H__inference_activation_7_layer_call_and_return_conditional_losses_182476

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:���������  �c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:���������  �"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������  �:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_8_layer_call_fn_184405

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_182174�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184401:&"
 
_user_specified_name184399:&"
 
_user_specified_name184397:&"
 
_user_specified_name184395:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_182192

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
H__inference_sequential_2_layer_call_and_return_conditional_losses_181298
random_flip_2_input"
random_flip_2_181068:	&
random_rotation_2_181189:	"
random_zoom_2_181294:	
identity��%random_flip_2/StatefulPartitionedCall�)random_rotation_2/StatefulPartitionedCall�%random_zoom_2/StatefulPartitionedCall�
%random_flip_2/StatefulPartitionedCallStatefulPartitionedCallrandom_flip_2_inputrandom_flip_2_181068*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_random_flip_2_layer_call_and_return_conditional_losses_181067�
)random_rotation_2/StatefulPartitionedCallStatefulPartitionedCall.random_flip_2/StatefulPartitionedCall:output:0random_rotation_2_181189*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_random_rotation_2_layer_call_and_return_conditional_losses_181188�
%random_zoom_2/StatefulPartitionedCallStatefulPartitionedCall2random_rotation_2/StatefulPartitionedCall:output:0random_zoom_2_181294*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_random_zoom_2_layer_call_and_return_conditional_losses_181293�
IdentityIdentity.random_zoom_2/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:������������
NoOpNoOp&^random_flip_2/StatefulPartitionedCall*^random_rotation_2/StatefulPartitionedCall&^random_zoom_2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:�����������: : : 2N
%random_flip_2/StatefulPartitionedCall%random_flip_2/StatefulPartitionedCall2V
)random_rotation_2/StatefulPartitionedCall)random_rotation_2/StatefulPartitionedCall2N
%random_zoom_2/StatefulPartitionedCall%random_zoom_2/StatefulPartitionedCall:&"
 
_user_specified_name181294:&"
 
_user_specified_name181189:&"
 
_user_specified_name181068:f b
1
_output_shapes
:�����������
-
_user_specified_namerandom_flip_2_input
�4
�
(__inference_model_2_layer_call_fn_183094
input_3!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: #
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: $

unknown_10: $

unknown_11:  $

unknown_12:  

unknown_13: 

unknown_14: 

unknown_15: 

unknown_16: $

unknown_17: $

unknown_18: @

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@$

unknown_23:@$

unknown_24:@@$

unknown_25: @

unknown_26:@

unknown_27:@

unknown_28:@

unknown_29:@$

unknown_30:@%

unknown_31:@�

unknown_32:	�

unknown_33:	�

unknown_34:	�

unknown_35:	�%

unknown_36:�&

unknown_37:��%

unknown_38:@�

unknown_39:	�

unknown_40:	�

unknown_41:	�

unknown_42:	�%

unknown_43:�&

unknown_44:��

unknown_45:	�

unknown_46:	�

unknown_47:	�

unknown_48:	�%

unknown_49:�&

unknown_50:��&

unknown_51:��

unknown_52:	�

unknown_53:	�

unknown_54:	�

unknown_55:	�%

unknown_56:�&

unknown_57:��

unknown_58:	�

unknown_59:	�

unknown_60:	�

unknown_61:	�%

unknown_62:�&

unknown_63:��&

unknown_64:��

unknown_65:	�

unknown_66:
identity��StatefulPartitionedCall�	
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65
unknown_66*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*f
_read_only_resource_inputsH
FD	
 !"#$%&'()*+,-./0123456789:;<=>?@ABCD*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_182806o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&D"
 
_user_specified_name183090:&C"
 
_user_specified_name183088:&B"
 
_user_specified_name183086:&A"
 
_user_specified_name183084:&@"
 
_user_specified_name183082:&?"
 
_user_specified_name183080:&>"
 
_user_specified_name183078:&="
 
_user_specified_name183076:&<"
 
_user_specified_name183074:&;"
 
_user_specified_name183072:&:"
 
_user_specified_name183070:&9"
 
_user_specified_name183068:&8"
 
_user_specified_name183066:&7"
 
_user_specified_name183064:&6"
 
_user_specified_name183062:&5"
 
_user_specified_name183060:&4"
 
_user_specified_name183058:&3"
 
_user_specified_name183056:&2"
 
_user_specified_name183054:&1"
 
_user_specified_name183052:&0"
 
_user_specified_name183050:&/"
 
_user_specified_name183048:&."
 
_user_specified_name183046:&-"
 
_user_specified_name183044:&,"
 
_user_specified_name183042:&+"
 
_user_specified_name183040:&*"
 
_user_specified_name183038:&)"
 
_user_specified_name183036:&("
 
_user_specified_name183034:&'"
 
_user_specified_name183032:&&"
 
_user_specified_name183030:&%"
 
_user_specified_name183028:&$"
 
_user_specified_name183026:&#"
 
_user_specified_name183024:&""
 
_user_specified_name183022:&!"
 
_user_specified_name183020:& "
 
_user_specified_name183018:&"
 
_user_specified_name183016:&"
 
_user_specified_name183014:&"
 
_user_specified_name183012:&"
 
_user_specified_name183010:&"
 
_user_specified_name183008:&"
 
_user_specified_name183006:&"
 
_user_specified_name183004:&"
 
_user_specified_name183002:&"
 
_user_specified_name183000:&"
 
_user_specified_name182998:&"
 
_user_specified_name182996:&"
 
_user_specified_name182994:&"
 
_user_specified_name182992:&"
 
_user_specified_name182990:&"
 
_user_specified_name182988:&"
 
_user_specified_name182986:&"
 
_user_specified_name182984:&"
 
_user_specified_name182982:&"
 
_user_specified_name182980:&"
 
_user_specified_name182978:&"
 
_user_specified_name182976:&
"
 
_user_specified_name182974:&	"
 
_user_specified_name182972:&"
 
_user_specified_name182970:&"
 
_user_specified_name182968:&"
 
_user_specified_name182966:&"
 
_user_specified_name182964:&"
 
_user_specified_name182962:&"
 
_user_specified_name182960:&"
 
_user_specified_name182958:&"
 
_user_specified_name182956:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_3
�
�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_181933

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_11_layer_call_and_return_conditional_losses_183621

inputs8
conv2d_readvariableop_resource:  
identity��Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������~~ *
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:���������~~ :
NoOpNoOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:����������� : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_181822

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
d
H__inference_activation_8_layer_call_and_return_conditional_losses_184464

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:����������c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
~
.__inference_random_zoom_2_layer_call_fn_184794

inputs
unknown:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_random_zoom_2_layer_call_and_return_conditional_losses_181293y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184790:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_181387

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+��������������������������� �
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+��������������������������� : : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�n
�
I__inference_random_zoom_2_layer_call_and_return_conditional_losses_181293

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity��stateful_uniform/RngReadAndSkipI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: Z
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :�
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *��L?Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *���?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:�
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :�
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:���������z
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: �
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:����������
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:���������M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2stateful_uniform:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������^
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
::��i
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
zoom_matrix/strided_sliceStridedSlicezoom_matrix/Shape:output:0(zoom_matrix/strided_slice/stack:output:0*zoom_matrix/strided_slice/stack_1:output:0*zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?_
zoom_matrix/subSub
Cast_1:y:0zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: Z
zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:���������x
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:���������X
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?a
zoom_matrix/sub_2SubCast:y:0zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: \
zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:���������|
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:���������v
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask\
zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
zoom_matrix/zeros/packedPack"zoom_matrix/strided_slice:output:0#zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:���������^
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
zoom_matrix/zeros_1/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������v
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask^
zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
zoom_matrix/zeros_2/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:���������Y
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������S
transform/ShapeShapeinputs*
T0*
_output_shapes
::��g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    �
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:�����������*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR�
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:�����������D
NoOpNoOp ^stateful_uniform/RngReadAndSkip*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:�����������: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:($
"
_user_specified_name
resource:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_181702

inputsB
(separable_conv2d_readvariableop_resource:@D
*separable_conv2d_readvariableop_1_resource:@@
identity��separable_conv2d/ReadVariableOp�!separable_conv2d/ReadVariableOp_1�
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:@@*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
�
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingVALID*
strides
�
IdentityIdentityseparable_conv2d:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@h
NoOpNoOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+���������������������������@: : 2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
e
I__inference_random_zoom_2_layer_call_and_return_conditional_losses_184905

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_182068

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_184559

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_14_layer_call_and_return_conditional_losses_184287

inputs:
conv2d_readvariableop_resource:��
identity��Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:����������:
NoOpNoOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:���������  �: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:X T
0
_output_shapes
:���������  �
 
_user_specified_nameinputs
�	
�
3__inference_separable_conv2d_4_layer_call_fn_184029

inputs"
unknown:�%
	unknown_0:��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_181878�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,����������������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184025:&"
 
_user_specified_name184023:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_9_layer_call_fn_184325

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_182109�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name184321:&"
 
_user_specified_name184319:&"
 
_user_specified_name184317:&"
 
_user_specified_name184315:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
p
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_182255

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_184121

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
d
H__inference_activation_2_layer_call_and_return_conditional_losses_183798

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������~~@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������~~@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������~~@:W S
/
_output_shapes
:���������~~@
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
input_3:
serving_default_input_3:0�����������;
dense_20
StatefulPartitionedCall:0���������tensorflow/serving/predict:��	
�
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer-17
layer_with_weights-9
layer-18
layer-19
layer_with_weights-10
layer-20
layer-21
layer_with_weights-11
layer-22
layer-23
layer_with_weights-12
layer-24
layer_with_weights-13
layer-25
layer-26
layer_with_weights-14
layer-27
layer-28
layer_with_weights-15
layer-29
layer-30
 layer_with_weights-16
 layer-31
!layer-32
"layer_with_weights-17
"layer-33
#layer_with_weights-18
#layer-34
$layer-35
%layer_with_weights-19
%layer-36
&layer-37
'layer_with_weights-20
'layer-38
(layer-39
)layer_with_weights-21
)layer-40
*layer-41
+layer_with_weights-22
+layer-42
,layer_with_weights-23
,layer-43
-layer-44
.layer_with_weights-24
.layer-45
/layer-46
0layer_with_weights-25
0layer-47
1layer-48
2layer-49
3layer-50
4layer_with_weights-26
4layer-51
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses
;_default_save_signature
<	optimizer
=
signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
>layer-0
?layer-1
@layer-2
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses

Skernel
 T_jit_compiled_convolution_op"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses
[axis
	\gamma
]beta
^moving_mean
_moving_variance"
_tf_keras_layer
�
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses"
_tf_keras_layer
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses
ldepthwise_kernel
mpointwise_kernel
 n_jit_compiled_convolution_op"
_tf_keras_layer
�
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses
uaxis
	vgamma
wbeta
xmoving_mean
ymoving_variance"
_tf_keras_layer
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
�pointwise_kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
�pointwise_kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
�pointwise_kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
�pointwise_kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
�pointwise_kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
�pointwise_kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
�pointwise_kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
�pointwise_kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
�pointwise_kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
S0
\1
]2
^3
_4
l5
m6
v7
w8
x9
y10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60
�61
�62
�63
�64
�65
�66
�67"
trackable_list_wrapper
�
S0
\1
]2
l3
m4
v5
w6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
;_default_save_signature
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_model_2_layer_call_fn_182953
(__inference_model_2_layer_call_fn_183094�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_model_2_layer_call_and_return_conditional_losses_182610
C__inference_model_2_layer_call_and_return_conditional_losses_182806�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�B�
!__inference__wrapped_model_181007input_3"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�_velocities
�
_momentums
�_average_gradients
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
-__inference_sequential_2_layer_call_fn_181328
-__inference_sequential_2_layer_call_fn_181333�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
H__inference_sequential_2_layer_call_and_return_conditional_losses_181298
H__inference_sequential_2_layer_call_and_return_conditional_losses_181317�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_rescaling_2_layer_call_fn_183389�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
G__inference_rescaling_2_layer_call_and_return_conditional_losses_183397�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
'
S0"
trackable_list_wrapper
'
S0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_10_layer_call_fn_183404�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_10_layer_call_and_return_conditional_losses_183411�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
*:( 2conv2d_10/kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
\0
]1
^2
_3"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_1_layer_call_fn_183424
6__inference_batch_normalization_1_layer_call_fn_183437�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_183455
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_183473�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
):' 2batch_normalization_1/gamma
(:& 2batch_normalization_1/beta
1:/  (2!batch_normalization_1/moving_mean
5:3  (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_activation_1_layer_call_fn_183478�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
H__inference_activation_1_layer_call_and_return_conditional_losses_183483�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_separable_conv2d_1_layer_call_fn_183492�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
N__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_183504�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
=:; 2#separable_conv2d_1/depthwise_kernel
=:;  2#separable_conv2d_1/pointwise_kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
v0
w1
x2
y3"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_batch_normalization_layer_call_fn_183517
4__inference_batch_normalization_layer_call_fn_183530�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_183548
O__inference_batch_normalization_layer_call_and_return_conditional_losses_183566�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
':% 2batch_normalization/gamma
&:$ 2batch_normalization/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_activation_layer_call_fn_183571�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
F__inference_activation_layer_call_and_return_conditional_losses_183576�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_separable_conv2d_layer_call_fn_183585�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
L__inference_separable_conv2d_layer_call_and_return_conditional_losses_183597�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
;:9 2!separable_conv2d/depthwise_kernel
;:9  2!separable_conv2d/pointwise_kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_max_pooling2d_8_layer_call_fn_183602�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_183607�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_11_layer_call_fn_183614�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_11_layer_call_and_return_conditional_losses_183621�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
*:(  2conv2d_11/kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
$__inference_add_layer_call_fn_183627�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
?__inference_add_layer_call_and_return_conditional_losses_183633�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_3_layer_call_fn_183646
6__inference_batch_normalization_3_layer_call_fn_183659�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_183677
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_183695�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
):' 2batch_normalization_3/gamma
(:& 2batch_normalization_3/beta
1:/  (2!batch_normalization_3/moving_mean
5:3  (2%batch_normalization_3/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_activation_3_layer_call_fn_183700�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
H__inference_activation_3_layer_call_and_return_conditional_losses_183705�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_separable_conv2d_3_layer_call_fn_183714�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_183726�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
=:; 2#separable_conv2d_3/depthwise_kernel
=:; @2#separable_conv2d_3/pointwise_kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_2_layer_call_fn_183739
6__inference_batch_normalization_2_layer_call_fn_183752�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_183770
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_183788�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_activation_2_layer_call_fn_183793�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
H__inference_activation_2_layer_call_and_return_conditional_losses_183798�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_separable_conv2d_2_layer_call_fn_183807�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_183819�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
=:;@2#separable_conv2d_2/depthwise_kernel
=:;@@2#separable_conv2d_2/pointwise_kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_max_pooling2d_9_layer_call_fn_183824�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_183829�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_12_layer_call_fn_183836�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_12_layer_call_and_return_conditional_losses_183843�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
*:( @2conv2d_12/kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_add_1_layer_call_fn_183849�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
A__inference_add_1_layer_call_and_return_conditional_losses_183855�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_5_layer_call_fn_183868
6__inference_batch_normalization_5_layer_call_fn_183881�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_183899
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_183917�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
):'@2batch_normalization_5/gamma
(:&@2batch_normalization_5/beta
1:/@ (2!batch_normalization_5/moving_mean
5:3@ (2%batch_normalization_5/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_activation_5_layer_call_fn_183922�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
H__inference_activation_5_layer_call_and_return_conditional_losses_183927�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_separable_conv2d_5_layer_call_fn_183936�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
N__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_183948�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
=:;@2#separable_conv2d_5/depthwise_kernel
>:<@�2#separable_conv2d_5/pointwise_kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_4_layer_call_fn_183961
6__inference_batch_normalization_4_layer_call_fn_183974�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_183992
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_184010�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(�2batch_normalization_4/gamma
):'�2batch_normalization_4/beta
2:0� (2!batch_normalization_4/moving_mean
6:4� (2%batch_normalization_4/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_activation_4_layer_call_fn_184015�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
H__inference_activation_4_layer_call_and_return_conditional_losses_184020�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_separable_conv2d_4_layer_call_fn_184029�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_184041�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
>:<�2#separable_conv2d_4/depthwise_kernel
?:=��2#separable_conv2d_4/pointwise_kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_10_layer_call_fn_184046�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_184051�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_13_layer_call_fn_184058�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_13_layer_call_and_return_conditional_losses_184065�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
+:)@�2conv2d_13/kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_add_2_layer_call_fn_184071�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
A__inference_add_2_layer_call_and_return_conditional_losses_184077�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_7_layer_call_fn_184090
6__inference_batch_normalization_7_layer_call_fn_184103�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_184121
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_184139�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(�2batch_normalization_7/gamma
):'�2batch_normalization_7/beta
2:0� (2!batch_normalization_7/moving_mean
6:4� (2%batch_normalization_7/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_activation_7_layer_call_fn_184144�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
H__inference_activation_7_layer_call_and_return_conditional_losses_184149�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_separable_conv2d_7_layer_call_fn_184158�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
N__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_184170�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
>:<�2#separable_conv2d_7/depthwise_kernel
?:=��2#separable_conv2d_7/pointwise_kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_6_layer_call_fn_184183
6__inference_batch_normalization_6_layer_call_fn_184196�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_184214
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_184232�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(�2batch_normalization_6/gamma
):'�2batch_normalization_6/beta
2:0� (2!batch_normalization_6/moving_mean
6:4� (2%batch_normalization_6/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_activation_6_layer_call_fn_184237�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
H__inference_activation_6_layer_call_and_return_conditional_losses_184242�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_separable_conv2d_6_layer_call_fn_184251�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
N__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_184263�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
>:<�2#separable_conv2d_6/depthwise_kernel
?:=��2#separable_conv2d_6/pointwise_kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_11_layer_call_fn_184268�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_184273�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_14_layer_call_fn_184280�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_14_layer_call_and_return_conditional_losses_184287�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
,:*��2conv2d_14/kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_add_3_layer_call_fn_184293�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
A__inference_add_3_layer_call_and_return_conditional_losses_184299�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_9_layer_call_fn_184312
6__inference_batch_normalization_9_layer_call_fn_184325�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_184343
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_184361�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(�2batch_normalization_9/gamma
):'�2batch_normalization_9/beta
2:0� (2!batch_normalization_9/moving_mean
6:4� (2%batch_normalization_9/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_activation_9_layer_call_fn_184366�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
H__inference_activation_9_layer_call_and_return_conditional_losses_184371�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_separable_conv2d_9_layer_call_fn_184380�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
N__inference_separable_conv2d_9_layer_call_and_return_conditional_losses_184392�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
>:<�2#separable_conv2d_9/depthwise_kernel
?:=��2#separable_conv2d_9/pointwise_kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_8_layer_call_fn_184405
6__inference_batch_normalization_8_layer_call_fn_184418�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_184436
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_184454�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(�2batch_normalization_8/gamma
):'�2batch_normalization_8/beta
2:0� (2!batch_normalization_8/moving_mean
6:4� (2%batch_normalization_8/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_activation_8_layer_call_fn_184459�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
H__inference_activation_8_layer_call_and_return_conditional_losses_184464�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_separable_conv2d_8_layer_call_fn_184473�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
N__inference_separable_conv2d_8_layer_call_and_return_conditional_losses_184485�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
>:<�2#separable_conv2d_8/depthwise_kernel
?:=��2#separable_conv2d_8/pointwise_kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_12_layer_call_fn_184490�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_184495�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_15_layer_call_fn_184502�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_15_layer_call_and_return_conditional_losses_184509�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
,:*��2conv2d_15/kernel
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_add_4_layer_call_fn_184515�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
A__inference_add_4_layer_call_and_return_conditional_losses_184521�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
9__inference_global_average_pooling2d_layer_call_fn_184526�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_184532�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_1_layer_call_fn_184537
*__inference_dropout_1_layer_call_fn_184542�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_1_layer_call_and_return_conditional_losses_184554
E__inference_dropout_1_layer_call_and_return_conditional_losses_184559�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_2_layer_call_fn_184568�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_2_layer_call_and_return_conditional_losses_184579�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
!:	�2dense_2/kernel
:2dense_2/bias
�
^0
_1
x2
y3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_model_2_layer_call_fn_182953input_3"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_2_layer_call_fn_183094input_3"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_2_layer_call_and_return_conditional_losses_182610input_3"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_2_layer_call_and_return_conditional_losses_182806input_3"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_183384input_3"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
	jinput_3
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
.__inference_random_flip_2_layer_call_fn_184586
.__inference_random_flip_2_layer_call_fn_184591�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
I__inference_random_flip_2_layer_call_and_return_conditional_losses_184649
I__inference_random_flip_2_layer_call_and_return_conditional_losses_184653�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
/
�
_generator"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
2__inference_random_rotation_2_layer_call_fn_184660
2__inference_random_rotation_2_layer_call_fn_184665�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
M__inference_random_rotation_2_layer_call_and_return_conditional_losses_184783
M__inference_random_rotation_2_layer_call_and_return_conditional_losses_184787�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
/
�
_generator"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
.__inference_random_zoom_2_layer_call_fn_184794
.__inference_random_zoom_2_layer_call_fn_184799�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
I__inference_random_zoom_2_layer_call_and_return_conditional_losses_184901
I__inference_random_zoom_2_layer_call_and_return_conditional_losses_184905�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
/
�
_generator"
_generic_user_object
 "
trackable_list_wrapper
5
>0
?1
@2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_sequential_2_layer_call_fn_181328random_flip_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_2_layer_call_fn_181333random_flip_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_2_layer_call_and_return_conditional_losses_181298random_flip_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_2_layer_call_and_return_conditional_losses_181317random_flip_2_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
,__inference_rescaling_2_layer_call_fn_183389inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
G__inference_rescaling_2_layer_call_and_return_conditional_losses_183397inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
*__inference_conv2d_10_layer_call_fn_183404inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_conv2d_10_layer_call_and_return_conditional_losses_183411inputs"�
���
FullArgSpec
args�

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
annotations� *
 
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_1_layer_call_fn_183424inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_1_layer_call_fn_183437inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_183455inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_183473inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
-__inference_activation_1_layer_call_fn_183478inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
H__inference_activation_1_layer_call_and_return_conditional_losses_183483inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
3__inference_separable_conv2d_1_layer_call_fn_183492inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
N__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_183504inputs"�
���
FullArgSpec
args�

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
annotations� *
 
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_batch_normalization_layer_call_fn_183517inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_batch_normalization_layer_call_fn_183530inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_183548inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_183566inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
+__inference_activation_layer_call_fn_183571inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
F__inference_activation_layer_call_and_return_conditional_losses_183576inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
1__inference_separable_conv2d_layer_call_fn_183585inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
L__inference_separable_conv2d_layer_call_and_return_conditional_losses_183597inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
0__inference_max_pooling2d_8_layer_call_fn_183602inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_183607inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
*__inference_conv2d_11_layer_call_fn_183614inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_conv2d_11_layer_call_and_return_conditional_losses_183621inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
$__inference_add_layer_call_fn_183627inputs_0inputs_1"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
?__inference_add_layer_call_and_return_conditional_losses_183633inputs_0inputs_1"�
���
FullArgSpec
args�

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
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_3_layer_call_fn_183646inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_3_layer_call_fn_183659inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_183677inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_183695inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
-__inference_activation_3_layer_call_fn_183700inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
H__inference_activation_3_layer_call_and_return_conditional_losses_183705inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
3__inference_separable_conv2d_3_layer_call_fn_183714inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_183726inputs"�
���
FullArgSpec
args�

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
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_2_layer_call_fn_183739inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_2_layer_call_fn_183752inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_183770inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_183788inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
-__inference_activation_2_layer_call_fn_183793inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
H__inference_activation_2_layer_call_and_return_conditional_losses_183798inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
3__inference_separable_conv2d_2_layer_call_fn_183807inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_183819inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
0__inference_max_pooling2d_9_layer_call_fn_183824inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_183829inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
*__inference_conv2d_12_layer_call_fn_183836inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_conv2d_12_layer_call_and_return_conditional_losses_183843inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
&__inference_add_1_layer_call_fn_183849inputs_0inputs_1"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
A__inference_add_1_layer_call_and_return_conditional_losses_183855inputs_0inputs_1"�
���
FullArgSpec
args�

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
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_5_layer_call_fn_183868inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_5_layer_call_fn_183881inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_183899inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_183917inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
-__inference_activation_5_layer_call_fn_183922inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
H__inference_activation_5_layer_call_and_return_conditional_losses_183927inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
3__inference_separable_conv2d_5_layer_call_fn_183936inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
N__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_183948inputs"�
���
FullArgSpec
args�

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
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_4_layer_call_fn_183961inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_4_layer_call_fn_183974inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_183992inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_184010inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
-__inference_activation_4_layer_call_fn_184015inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
H__inference_activation_4_layer_call_and_return_conditional_losses_184020inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
3__inference_separable_conv2d_4_layer_call_fn_184029inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_184041inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
1__inference_max_pooling2d_10_layer_call_fn_184046inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_184051inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
*__inference_conv2d_13_layer_call_fn_184058inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_conv2d_13_layer_call_and_return_conditional_losses_184065inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
&__inference_add_2_layer_call_fn_184071inputs_0inputs_1"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
A__inference_add_2_layer_call_and_return_conditional_losses_184077inputs_0inputs_1"�
���
FullArgSpec
args�

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
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_7_layer_call_fn_184090inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_7_layer_call_fn_184103inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_184121inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_184139inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
-__inference_activation_7_layer_call_fn_184144inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
H__inference_activation_7_layer_call_and_return_conditional_losses_184149inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
3__inference_separable_conv2d_7_layer_call_fn_184158inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
N__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_184170inputs"�
���
FullArgSpec
args�

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
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_6_layer_call_fn_184183inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_6_layer_call_fn_184196inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_184214inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_184232inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
-__inference_activation_6_layer_call_fn_184237inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
H__inference_activation_6_layer_call_and_return_conditional_losses_184242inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
3__inference_separable_conv2d_6_layer_call_fn_184251inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
N__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_184263inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
1__inference_max_pooling2d_11_layer_call_fn_184268inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_184273inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
*__inference_conv2d_14_layer_call_fn_184280inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_conv2d_14_layer_call_and_return_conditional_losses_184287inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
&__inference_add_3_layer_call_fn_184293inputs_0inputs_1"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
A__inference_add_3_layer_call_and_return_conditional_losses_184299inputs_0inputs_1"�
���
FullArgSpec
args�

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
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_9_layer_call_fn_184312inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_9_layer_call_fn_184325inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_184343inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_184361inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
-__inference_activation_9_layer_call_fn_184366inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
H__inference_activation_9_layer_call_and_return_conditional_losses_184371inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
3__inference_separable_conv2d_9_layer_call_fn_184380inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
N__inference_separable_conv2d_9_layer_call_and_return_conditional_losses_184392inputs"�
���
FullArgSpec
args�

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
annotations� *
 
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_8_layer_call_fn_184405inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_8_layer_call_fn_184418inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_184436inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_184454inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
-__inference_activation_8_layer_call_fn_184459inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
H__inference_activation_8_layer_call_and_return_conditional_losses_184464inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
3__inference_separable_conv2d_8_layer_call_fn_184473inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
N__inference_separable_conv2d_8_layer_call_and_return_conditional_losses_184485inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
1__inference_max_pooling2d_12_layer_call_fn_184490inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_184495inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
*__inference_conv2d_15_layer_call_fn_184502inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_conv2d_15_layer_call_and_return_conditional_losses_184509inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
&__inference_add_4_layer_call_fn_184515inputs_0inputs_1"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
A__inference_add_4_layer_call_and_return_conditional_losses_184521inputs_0inputs_1"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
9__inference_global_average_pooling2d_layer_call_fn_184526inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_184532inputs"�
���
FullArgSpec
args�

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
annotations� *
 
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
�B�
*__inference_dropout_1_layer_call_fn_184537inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_1_layer_call_fn_184542inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_1_layer_call_and_return_conditional_losses_184554inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_1_layer_call_and_return_conditional_losses_184559inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_dense_2_layer_call_fn_184568inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
C__inference_dense_2_layer_call_and_return_conditional_losses_184579inputs"�
���
FullArgSpec
args�

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
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
9:7 2!RMSprop/velocity/conv2d_10/kernel
8:6 2,RMSprop/velocity/batch_normalization_1/gamma
7:5 2+RMSprop/velocity/batch_normalization_1/beta
L:J 24RMSprop/velocity/separable_conv2d_1/depthwise_kernel
L:J  24RMSprop/velocity/separable_conv2d_1/pointwise_kernel
6:4 2*RMSprop/velocity/batch_normalization/gamma
5:3 2)RMSprop/velocity/batch_normalization/beta
J:H 22RMSprop/velocity/separable_conv2d/depthwise_kernel
J:H  22RMSprop/velocity/separable_conv2d/pointwise_kernel
9:7  2!RMSprop/velocity/conv2d_11/kernel
8:6 2,RMSprop/velocity/batch_normalization_3/gamma
7:5 2+RMSprop/velocity/batch_normalization_3/beta
L:J 24RMSprop/velocity/separable_conv2d_3/depthwise_kernel
L:J @24RMSprop/velocity/separable_conv2d_3/pointwise_kernel
8:6@2,RMSprop/velocity/batch_normalization_2/gamma
7:5@2+RMSprop/velocity/batch_normalization_2/beta
L:J@24RMSprop/velocity/separable_conv2d_2/depthwise_kernel
L:J@@24RMSprop/velocity/separable_conv2d_2/pointwise_kernel
9:7 @2!RMSprop/velocity/conv2d_12/kernel
8:6@2,RMSprop/velocity/batch_normalization_5/gamma
7:5@2+RMSprop/velocity/batch_normalization_5/beta
L:J@24RMSprop/velocity/separable_conv2d_5/depthwise_kernel
M:K@�24RMSprop/velocity/separable_conv2d_5/pointwise_kernel
9:7�2,RMSprop/velocity/batch_normalization_4/gamma
8:6�2+RMSprop/velocity/batch_normalization_4/beta
M:K�24RMSprop/velocity/separable_conv2d_4/depthwise_kernel
N:L��24RMSprop/velocity/separable_conv2d_4/pointwise_kernel
::8@�2!RMSprop/velocity/conv2d_13/kernel
9:7�2,RMSprop/velocity/batch_normalization_7/gamma
8:6�2+RMSprop/velocity/batch_normalization_7/beta
M:K�24RMSprop/velocity/separable_conv2d_7/depthwise_kernel
N:L��24RMSprop/velocity/separable_conv2d_7/pointwise_kernel
9:7�2,RMSprop/velocity/batch_normalization_6/gamma
8:6�2+RMSprop/velocity/batch_normalization_6/beta
M:K�24RMSprop/velocity/separable_conv2d_6/depthwise_kernel
N:L��24RMSprop/velocity/separable_conv2d_6/pointwise_kernel
;:9��2!RMSprop/velocity/conv2d_14/kernel
9:7�2,RMSprop/velocity/batch_normalization_9/gamma
8:6�2+RMSprop/velocity/batch_normalization_9/beta
M:K�24RMSprop/velocity/separable_conv2d_9/depthwise_kernel
N:L��24RMSprop/velocity/separable_conv2d_9/pointwise_kernel
9:7�2,RMSprop/velocity/batch_normalization_8/gamma
8:6�2+RMSprop/velocity/batch_normalization_8/beta
M:K�24RMSprop/velocity/separable_conv2d_8/depthwise_kernel
N:L��24RMSprop/velocity/separable_conv2d_8/pointwise_kernel
;:9��2!RMSprop/velocity/conv2d_15/kernel
0:.	�2RMSprop/velocity/dense_2/kernel
):'2RMSprop/velocity/dense_2/bias
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
�B�
.__inference_random_flip_2_layer_call_fn_184586inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_random_flip_2_layer_call_fn_184591inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_random_flip_2_layer_call_and_return_conditional_losses_184649inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_random_flip_2_layer_call_and_return_conditional_losses_184653inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
/
�
_state_var"
_generic_user_object
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
�B�
2__inference_random_rotation_2_layer_call_fn_184660inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
2__inference_random_rotation_2_layer_call_fn_184665inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_random_rotation_2_layer_call_and_return_conditional_losses_184783inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_random_rotation_2_layer_call_and_return_conditional_losses_184787inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
/
�
_state_var"
_generic_user_object
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
�B�
.__inference_random_zoom_2_layer_call_fn_184794inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_random_zoom_2_layer_call_fn_184799inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_random_zoom_2_layer_call_and_return_conditional_losses_184901inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_random_zoom_2_layer_call_and_return_conditional_losses_184905inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
/
�
_state_var"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
": 	2random_flip_2/StateVar
&:$	2random_rotation_2/StateVar
": 	2random_zoom_2/StateVar�
!__inference__wrapped_model_181007�}S\]^_lmvwxy���������������������������������������������������������:�7
0�-
+�(
input_3�����������
� "1�.
,
dense_2!�
dense_2����������
H__inference_activation_1_layer_call_and_return_conditional_losses_183483s9�6
/�,
*�'
inputs����������� 
� "6�3
,�)
tensor_0����������� 
� �
-__inference_activation_1_layer_call_fn_183478h9�6
/�,
*�'
inputs����������� 
� "+�(
unknown����������� �
H__inference_activation_2_layer_call_and_return_conditional_losses_183798o7�4
-�*
(�%
inputs���������~~@
� "4�1
*�'
tensor_0���������~~@
� �
-__inference_activation_2_layer_call_fn_183793d7�4
-�*
(�%
inputs���������~~@
� ")�&
unknown���������~~@�
H__inference_activation_3_layer_call_and_return_conditional_losses_183705o7�4
-�*
(�%
inputs���������~~ 
� "4�1
*�'
tensor_0���������~~ 
� �
-__inference_activation_3_layer_call_fn_183700d7�4
-�*
(�%
inputs���������~~ 
� ")�&
unknown���������~~ �
H__inference_activation_4_layer_call_and_return_conditional_losses_184020q8�5
.�+
)�&
inputs���������??�
� "5�2
+�(
tensor_0���������??�
� �
-__inference_activation_4_layer_call_fn_184015f8�5
.�+
)�&
inputs���������??�
� "*�'
unknown���������??��
H__inference_activation_5_layer_call_and_return_conditional_losses_183927o7�4
-�*
(�%
inputs���������??@
� "4�1
*�'
tensor_0���������??@
� �
-__inference_activation_5_layer_call_fn_183922d7�4
-�*
(�%
inputs���������??@
� ")�&
unknown���������??@�
H__inference_activation_6_layer_call_and_return_conditional_losses_184242q8�5
.�+
)�&
inputs���������  �
� "5�2
+�(
tensor_0���������  �
� �
-__inference_activation_6_layer_call_fn_184237f8�5
.�+
)�&
inputs���������  �
� "*�'
unknown���������  ��
H__inference_activation_7_layer_call_and_return_conditional_losses_184149q8�5
.�+
)�&
inputs���������  �
� "5�2
+�(
tensor_0���������  �
� �
-__inference_activation_7_layer_call_fn_184144f8�5
.�+
)�&
inputs���������  �
� "*�'
unknown���������  ��
H__inference_activation_8_layer_call_and_return_conditional_losses_184464q8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
-__inference_activation_8_layer_call_fn_184459f8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
H__inference_activation_9_layer_call_and_return_conditional_losses_184371q8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
-__inference_activation_9_layer_call_fn_184366f8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
F__inference_activation_layer_call_and_return_conditional_losses_183576s9�6
/�,
*�'
inputs����������� 
� "6�3
,�)
tensor_0����������� 
� �
+__inference_activation_layer_call_fn_183571h9�6
/�,
*�'
inputs����������� 
� "+�(
unknown����������� �
A__inference_add_1_layer_call_and_return_conditional_losses_183855�j�g
`�]
[�X
*�'
inputs_0���������??@
*�'
inputs_1���������??@
� "4�1
*�'
tensor_0���������??@
� �
&__inference_add_1_layer_call_fn_183849�j�g
`�]
[�X
*�'
inputs_0���������??@
*�'
inputs_1���������??@
� ")�&
unknown���������??@�
A__inference_add_2_layer_call_and_return_conditional_losses_184077�l�i
b�_
]�Z
+�(
inputs_0���������  �
+�(
inputs_1���������  �
� "5�2
+�(
tensor_0���������  �
� �
&__inference_add_2_layer_call_fn_184071�l�i
b�_
]�Z
+�(
inputs_0���������  �
+�(
inputs_1���������  �
� "*�'
unknown���������  ��
A__inference_add_3_layer_call_and_return_conditional_losses_184299�l�i
b�_
]�Z
+�(
inputs_0����������
+�(
inputs_1����������
� "5�2
+�(
tensor_0����������
� �
&__inference_add_3_layer_call_fn_184293�l�i
b�_
]�Z
+�(
inputs_0����������
+�(
inputs_1����������
� "*�'
unknown�����������
A__inference_add_4_layer_call_and_return_conditional_losses_184521�l�i
b�_
]�Z
+�(
inputs_0����������
+�(
inputs_1����������
� "5�2
+�(
tensor_0����������
� �
&__inference_add_4_layer_call_fn_184515�l�i
b�_
]�Z
+�(
inputs_0����������
+�(
inputs_1����������
� "*�'
unknown�����������
?__inference_add_layer_call_and_return_conditional_losses_183633�j�g
`�]
[�X
*�'
inputs_0���������~~ 
*�'
inputs_1���������~~ 
� "4�1
*�'
tensor_0���������~~ 
� �
$__inference_add_layer_call_fn_183627�j�g
`�]
[�X
*�'
inputs_0���������~~ 
*�'
inputs_1���������~~ 
� ")�&
unknown���������~~ �
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_183455�\]^_Q�N
G�D
:�7
inputs+��������������������������� 
p

 
� "F�C
<�9
tensor_0+��������������������������� 
� �
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_183473�\]^_Q�N
G�D
:�7
inputs+��������������������������� 
p 

 
� "F�C
<�9
tensor_0+��������������������������� 
� �
6__inference_batch_normalization_1_layer_call_fn_183424�\]^_Q�N
G�D
:�7
inputs+��������������������������� 
p

 
� ";�8
unknown+��������������������������� �
6__inference_batch_normalization_1_layer_call_fn_183437�\]^_Q�N
G�D
:�7
inputs+��������������������������� 
p 

 
� ";�8
unknown+��������������������������� �
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_183770�����Q�N
G�D
:�7
inputs+���������������������������@
p

 
� "F�C
<�9
tensor_0+���������������������������@
� �
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_183788�����Q�N
G�D
:�7
inputs+���������������������������@
p 

 
� "F�C
<�9
tensor_0+���������������������������@
� �
6__inference_batch_normalization_2_layer_call_fn_183739�����Q�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
6__inference_batch_normalization_2_layer_call_fn_183752�����Q�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_183677�����Q�N
G�D
:�7
inputs+��������������������������� 
p

 
� "F�C
<�9
tensor_0+��������������������������� 
� �
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_183695�����Q�N
G�D
:�7
inputs+��������������������������� 
p 

 
� "F�C
<�9
tensor_0+��������������������������� 
� �
6__inference_batch_normalization_3_layer_call_fn_183646�����Q�N
G�D
:�7
inputs+��������������������������� 
p

 
� ";�8
unknown+��������������������������� �
6__inference_batch_normalization_3_layer_call_fn_183659�����Q�N
G�D
:�7
inputs+��������������������������� 
p 

 
� ";�8
unknown+��������������������������� �
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_183992�����R�O
H�E
;�8
inputs,����������������������������
p

 
� "G�D
=�:
tensor_0,����������������������������
� �
Q__inference_batch_normalization_4_layer_call_and_return_conditional_losses_184010�����R�O
H�E
;�8
inputs,����������������������������
p 

 
� "G�D
=�:
tensor_0,����������������������������
� �
6__inference_batch_normalization_4_layer_call_fn_183961�����R�O
H�E
;�8
inputs,����������������������������
p

 
� "<�9
unknown,�����������������������������
6__inference_batch_normalization_4_layer_call_fn_183974�����R�O
H�E
;�8
inputs,����������������������������
p 

 
� "<�9
unknown,�����������������������������
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_183899�����Q�N
G�D
:�7
inputs+���������������������������@
p

 
� "F�C
<�9
tensor_0+���������������������������@
� �
Q__inference_batch_normalization_5_layer_call_and_return_conditional_losses_183917�����Q�N
G�D
:�7
inputs+���������������������������@
p 

 
� "F�C
<�9
tensor_0+���������������������������@
� �
6__inference_batch_normalization_5_layer_call_fn_183868�����Q�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
6__inference_batch_normalization_5_layer_call_fn_183881�����Q�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_184214�����R�O
H�E
;�8
inputs,����������������������������
p

 
� "G�D
=�:
tensor_0,����������������������������
� �
Q__inference_batch_normalization_6_layer_call_and_return_conditional_losses_184232�����R�O
H�E
;�8
inputs,����������������������������
p 

 
� "G�D
=�:
tensor_0,����������������������������
� �
6__inference_batch_normalization_6_layer_call_fn_184183�����R�O
H�E
;�8
inputs,����������������������������
p

 
� "<�9
unknown,�����������������������������
6__inference_batch_normalization_6_layer_call_fn_184196�����R�O
H�E
;�8
inputs,����������������������������
p 

 
� "<�9
unknown,�����������������������������
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_184121�����R�O
H�E
;�8
inputs,����������������������������
p

 
� "G�D
=�:
tensor_0,����������������������������
� �
Q__inference_batch_normalization_7_layer_call_and_return_conditional_losses_184139�����R�O
H�E
;�8
inputs,����������������������������
p 

 
� "G�D
=�:
tensor_0,����������������������������
� �
6__inference_batch_normalization_7_layer_call_fn_184090�����R�O
H�E
;�8
inputs,����������������������������
p

 
� "<�9
unknown,�����������������������������
6__inference_batch_normalization_7_layer_call_fn_184103�����R�O
H�E
;�8
inputs,����������������������������
p 

 
� "<�9
unknown,�����������������������������
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_184436�����R�O
H�E
;�8
inputs,����������������������������
p

 
� "G�D
=�:
tensor_0,����������������������������
� �
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_184454�����R�O
H�E
;�8
inputs,����������������������������
p 

 
� "G�D
=�:
tensor_0,����������������������������
� �
6__inference_batch_normalization_8_layer_call_fn_184405�����R�O
H�E
;�8
inputs,����������������������������
p

 
� "<�9
unknown,�����������������������������
6__inference_batch_normalization_8_layer_call_fn_184418�����R�O
H�E
;�8
inputs,����������������������������
p 

 
� "<�9
unknown,�����������������������������
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_184343�����R�O
H�E
;�8
inputs,����������������������������
p

 
� "G�D
=�:
tensor_0,����������������������������
� �
Q__inference_batch_normalization_9_layer_call_and_return_conditional_losses_184361�����R�O
H�E
;�8
inputs,����������������������������
p 

 
� "G�D
=�:
tensor_0,����������������������������
� �
6__inference_batch_normalization_9_layer_call_fn_184312�����R�O
H�E
;�8
inputs,����������������������������
p

 
� "<�9
unknown,�����������������������������
6__inference_batch_normalization_9_layer_call_fn_184325�����R�O
H�E
;�8
inputs,����������������������������
p 

 
� "<�9
unknown,�����������������������������
O__inference_batch_normalization_layer_call_and_return_conditional_losses_183548�vwxyQ�N
G�D
:�7
inputs+��������������������������� 
p

 
� "F�C
<�9
tensor_0+��������������������������� 
� �
O__inference_batch_normalization_layer_call_and_return_conditional_losses_183566�vwxyQ�N
G�D
:�7
inputs+��������������������������� 
p 

 
� "F�C
<�9
tensor_0+��������������������������� 
� �
4__inference_batch_normalization_layer_call_fn_183517�vwxyQ�N
G�D
:�7
inputs+��������������������������� 
p

 
� ";�8
unknown+��������������������������� �
4__inference_batch_normalization_layer_call_fn_183530�vwxyQ�N
G�D
:�7
inputs+��������������������������� 
p 

 
� ";�8
unknown+��������������������������� �
E__inference_conv2d_10_layer_call_and_return_conditional_losses_183411vS9�6
/�,
*�'
inputs�����������
� "6�3
,�)
tensor_0����������� 
� �
*__inference_conv2d_10_layer_call_fn_183404kS9�6
/�,
*�'
inputs�����������
� "+�(
unknown����������� �
E__inference_conv2d_11_layer_call_and_return_conditional_losses_183621u�9�6
/�,
*�'
inputs����������� 
� "4�1
*�'
tensor_0���������~~ 
� �
*__inference_conv2d_11_layer_call_fn_183614j�9�6
/�,
*�'
inputs����������� 
� ")�&
unknown���������~~ �
E__inference_conv2d_12_layer_call_and_return_conditional_losses_183843s�7�4
-�*
(�%
inputs���������~~ 
� "4�1
*�'
tensor_0���������??@
� �
*__inference_conv2d_12_layer_call_fn_183836h�7�4
-�*
(�%
inputs���������~~ 
� ")�&
unknown���������??@�
E__inference_conv2d_13_layer_call_and_return_conditional_losses_184065t�7�4
-�*
(�%
inputs���������??@
� "5�2
+�(
tensor_0���������  �
� �
*__inference_conv2d_13_layer_call_fn_184058i�7�4
-�*
(�%
inputs���������??@
� "*�'
unknown���������  ��
E__inference_conv2d_14_layer_call_and_return_conditional_losses_184287u�8�5
.�+
)�&
inputs���������  �
� "5�2
+�(
tensor_0����������
� �
*__inference_conv2d_14_layer_call_fn_184280j�8�5
.�+
)�&
inputs���������  �
� "*�'
unknown�����������
E__inference_conv2d_15_layer_call_and_return_conditional_losses_184509u�8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
*__inference_conv2d_15_layer_call_fn_184502j�8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
C__inference_dense_2_layer_call_and_return_conditional_losses_184579f��0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
(__inference_dense_2_layer_call_fn_184568[��0�-
&�#
!�
inputs����������
� "!�
unknown����������
E__inference_dropout_1_layer_call_and_return_conditional_losses_184554e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
E__inference_dropout_1_layer_call_and_return_conditional_losses_184559e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
*__inference_dropout_1_layer_call_fn_184537Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
*__inference_dropout_1_layer_call_fn_184542Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_184532�R�O
H�E
C�@
inputs4������������������������������������
� "5�2
+�(
tensor_0������������������
� �
9__inference_global_average_pooling2d_layer_call_fn_184526�R�O
H�E
C�@
inputs4������������������������������������
� "*�'
unknown�������������������
L__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_184051�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_max_pooling2d_10_layer_call_fn_184046�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
L__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_184273�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_max_pooling2d_11_layer_call_fn_184268�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
L__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_184495�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_max_pooling2d_12_layer_call_fn_184490�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
K__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_183607�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
0__inference_max_pooling2d_8_layer_call_fn_183602�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
K__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_183829�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
0__inference_max_pooling2d_9_layer_call_fn_183824�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
C__inference_model_2_layer_call_and_return_conditional_losses_182610�����S\]^_lmvwxy���������������������������������������������������������B�?
8�5
+�(
input_3�����������
p

 
� ",�)
"�
tensor_0���������
� �
C__inference_model_2_layer_call_and_return_conditional_losses_182806�}S\]^_lmvwxy���������������������������������������������������������B�?
8�5
+�(
input_3�����������
p 

 
� ",�)
"�
tensor_0���������
� �
(__inference_model_2_layer_call_fn_182953�����S\]^_lmvwxy���������������������������������������������������������B�?
8�5
+�(
input_3�����������
p

 
� "!�
unknown����������
(__inference_model_2_layer_call_fn_183094�}S\]^_lmvwxy���������������������������������������������������������B�?
8�5
+�(
input_3�����������
p 

 
� "!�
unknown����������
I__inference_random_flip_2_layer_call_and_return_conditional_losses_184649{�=�:
3�0
*�'
inputs�����������
p
� "6�3
,�)
tensor_0�����������
� �
I__inference_random_flip_2_layer_call_and_return_conditional_losses_184653w=�:
3�0
*�'
inputs�����������
p 
� "6�3
,�)
tensor_0�����������
� �
.__inference_random_flip_2_layer_call_fn_184586p�=�:
3�0
*�'
inputs�����������
p
� "+�(
unknown������������
.__inference_random_flip_2_layer_call_fn_184591l=�:
3�0
*�'
inputs�����������
p 
� "+�(
unknown������������
M__inference_random_rotation_2_layer_call_and_return_conditional_losses_184783{�=�:
3�0
*�'
inputs�����������
p
� "6�3
,�)
tensor_0�����������
� �
M__inference_random_rotation_2_layer_call_and_return_conditional_losses_184787w=�:
3�0
*�'
inputs�����������
p 
� "6�3
,�)
tensor_0�����������
� �
2__inference_random_rotation_2_layer_call_fn_184660p�=�:
3�0
*�'
inputs�����������
p
� "+�(
unknown������������
2__inference_random_rotation_2_layer_call_fn_184665l=�:
3�0
*�'
inputs�����������
p 
� "+�(
unknown������������
I__inference_random_zoom_2_layer_call_and_return_conditional_losses_184901{�=�:
3�0
*�'
inputs�����������
p
� "6�3
,�)
tensor_0�����������
� �
I__inference_random_zoom_2_layer_call_and_return_conditional_losses_184905w=�:
3�0
*�'
inputs�����������
p 
� "6�3
,�)
tensor_0�����������
� �
.__inference_random_zoom_2_layer_call_fn_184794p�=�:
3�0
*�'
inputs�����������
p
� "+�(
unknown������������
.__inference_random_zoom_2_layer_call_fn_184799l=�:
3�0
*�'
inputs�����������
p 
� "+�(
unknown������������
G__inference_rescaling_2_layer_call_and_return_conditional_losses_183397s9�6
/�,
*�'
inputs�����������
� "6�3
,�)
tensor_0�����������
� �
,__inference_rescaling_2_layer_call_fn_183389h9�6
/�,
*�'
inputs�����������
� "+�(
unknown������������
N__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_183504�lmI�F
?�<
:�7
inputs+��������������������������� 
� "F�C
<�9
tensor_0+��������������������������� 
� �
3__inference_separable_conv2d_1_layer_call_fn_183492�lmI�F
?�<
:�7
inputs+��������������������������� 
� ";�8
unknown+��������������������������� �
N__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_183819���I�F
?�<
:�7
inputs+���������������������������@
� "F�C
<�9
tensor_0+���������������������������@
� �
3__inference_separable_conv2d_2_layer_call_fn_183807���I�F
?�<
:�7
inputs+���������������������������@
� ";�8
unknown+���������������������������@�
N__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_183726���I�F
?�<
:�7
inputs+��������������������������� 
� "F�C
<�9
tensor_0+���������������������������@
� �
3__inference_separable_conv2d_3_layer_call_fn_183714���I�F
?�<
:�7
inputs+��������������������������� 
� ";�8
unknown+���������������������������@�
N__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_184041���J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
3__inference_separable_conv2d_4_layer_call_fn_184029���J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
N__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_183948���I�F
?�<
:�7
inputs+���������������������������@
� "G�D
=�:
tensor_0,����������������������������
� �
3__inference_separable_conv2d_5_layer_call_fn_183936���I�F
?�<
:�7
inputs+���������������������������@
� "<�9
unknown,�����������������������������
N__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_184263���J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
3__inference_separable_conv2d_6_layer_call_fn_184251���J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
N__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_184170���J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
3__inference_separable_conv2d_7_layer_call_fn_184158���J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
N__inference_separable_conv2d_8_layer_call_and_return_conditional_losses_184485���J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
3__inference_separable_conv2d_8_layer_call_fn_184473���J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
N__inference_separable_conv2d_9_layer_call_and_return_conditional_losses_184392���J�G
@�=
;�8
inputs,����������������������������
� "G�D
=�:
tensor_0,����������������������������
� �
3__inference_separable_conv2d_9_layer_call_fn_184380���J�G
@�=
;�8
inputs,����������������������������
� "<�9
unknown,�����������������������������
L__inference_separable_conv2d_layer_call_and_return_conditional_losses_183597���I�F
?�<
:�7
inputs+��������������������������� 
� "F�C
<�9
tensor_0+��������������������������� 
� �
1__inference_separable_conv2d_layer_call_fn_183585���I�F
?�<
:�7
inputs+��������������������������� 
� ";�8
unknown+��������������������������� �
H__inference_sequential_2_layer_call_and_return_conditional_losses_181298����N�K
D�A
7�4
random_flip_2_input�����������
p

 
� "6�3
,�)
tensor_0�����������
� �
H__inference_sequential_2_layer_call_and_return_conditional_losses_181317�N�K
D�A
7�4
random_flip_2_input�����������
p 

 
� "6�3
,�)
tensor_0�����������
� �
-__inference_sequential_2_layer_call_fn_181328����N�K
D�A
7�4
random_flip_2_input�����������
p

 
� "+�(
unknown������������
-__inference_sequential_2_layer_call_fn_181333}N�K
D�A
7�4
random_flip_2_input�����������
p 

 
� "+�(
unknown������������
$__inference_signature_wrapper_183384�}S\]^_lmvwxy���������������������������������������������������������E�B
� 
;�8
6
input_3+�(
input_3�����������"1�.
,
dense_2!�
dense_2���������