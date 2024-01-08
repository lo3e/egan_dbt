from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, \
    batch_norm, DropoutLayer, Deconv2DLayer, BatchNormLayer,\
    NonlinearityLayer, ElemwiseSumLayer, ConcatLayer, FlattenLayer,\
    Pool2DLayer, Upscale2DLayer, Conv2DLayer
from lasagne.nonlinearities import sigmoid, LeakyRectify, sigmoid,\
    tanh, softmax, elu
from lasagne.init import Normal, HeNormal

def build_generator_256(noise=None, ngf=128):

    lrelu = LeakyRectify(0.2)

    #noise input
    InputNoise = InputLayer(shape=(None, 100), input_var=noise)
    #FC Layer
    gnet0 = DenseLayer(InputNoise, ngf*32*4*4, W=Normal(0.02), nonlinearity=lrelu)
    print("Gen fc1:", gnet0.output_shape)
    #Reshape Layer
    gnet1 = ReshapeLayer(gnet0, ([0], ngf*32, 4, 4))
    print("Gen rs1:", gnet1.output_shape)
    #DeConv Layer
    gnet2 = Deconv2DLayer(gnet1, ngf*16, (4, 4), (2, 2), crop=1,
                          W=Normal(0.02), nonlinearity=lrelu)
    print("Gen deconv1:", gnet2.output_shape)
    #DeConv Layer
    gnet3 = Deconv2DLayer(gnet2, ngf*16, (4, 4), (2, 2), crop=1,
                          W=Normal(0.02), nonlinearity=lrelu)
    print("Gen deconv2:", gnet3.output_shape)
    #DeConv Layer
    gnet4 = Deconv2DLayer(gnet3, ngf*8, (4, 4), (2, 2), crop=1,
                          W=Normal(0.02), nonlinearity=lrelu)
    print("Gen deconv3:", gnet4.output_shape)
    #DeConv Layer
    gnet5 = Deconv2DLayer(gnet4, ngf*8, (4, 4), (2, 2), crop=1,
                          W=Normal(0.02), nonlinearity=lrelu)
    print("Gen deconv4:", gnet5.output_shape)
    #DeConv Layer
    gnet6 = Deconv2DLayer(gnet5, ngf*4, (4, 4), (2, 2), crop=1,
                          W=Normal(0.02), nonlinearity=lrelu)
    print("Gen deconv5:", gnet6.output_shape)
    #DeConv Layer
    gnet7 = Deconv2DLayer(gnet6, ngf*2, (4, 4), (2, 2), crop=1,
                          W=Normal(0.02), nonlinearity=lrelu)
    print("Gen deconv6:", gnet7.output_shape)
    #DeConv Layer
    gnet8 = Deconv2DLayer(gnet7, 1, (3, 3), (1, 1), crop='same',
                          W=Normal(0.02), nonlinearity=tanh)
    print("Gen output:", gnet8.output_shape)

    return gnet8

def build_discriminator_256(image=None, ndf=128):

    lrelu = LeakyRectify(0.2)

    #input images
    InputImg = InputLayer(shape=(None, 1, 256, 256), input_var=image)
    print("Dis Img_input:", InputImg.output_shape)
    #Conv Layer
    dis1 = Conv2DLayer(InputImg, ndf, (4, 4), (2, 2), pad=1,
                       W=Normal(0.02), nonlinearity=lrelu)
    print("Dis conv1:", dis1.output_shape)
    #Conv Layer
    dis2 = batch_norm(Conv2DLayer(dis1, ndf*2, (4, 4), (2, 2), pad=1,
                                  W=Normal(0.02), nonlinearity=lrelu))
    print("Dis conv2:", dis2.output_shape)
    #Conv Layer
    dis3 = batch_norm(Conv2DLayer(dis2, ndf*4, (4, 4), (2, 2), pad=1,
                                  W=Normal(0.02), nonlinearity=lrelu))
    print("Dis conv3:", dis3.output_shape)
    #Conv Layer
    dis4 = batch_norm(Conv2DLayer(dis3, ndf*8, (4, 4), (2, 2), pad=1,
                                  W=Normal(0.02), nonlinearity=lrelu))
    print("Dis conv3:", dis4.output_shape)
    #Conv Layer
    dis5 = batch_norm(Conv2DLayer(dis4, ndf*16, (4, 4), (2, 2), pad=1,
                                  W=Normal(0.02), nonlinearity=lrelu))
    print("Dis conv4:", dis5.output_shape)
    #Conv Layer
    dis6 = batch_norm(Conv2DLayer(dis4, ndf*32, (4, 4), (2, 2), pad=1,
                                  W=Normal(0.02), nonlinearity=lrelu))
    print("Dis conv5:", dis6.output_shape)
    #Conv Layer
    dis7 = DenseLayer(dis6, 1, W=Normal(0.02), nonlinearity=sigmoid)
    print("Dis output:", dis7.output_shape)

    return dis7