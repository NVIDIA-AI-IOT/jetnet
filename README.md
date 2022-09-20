# JetNet

<img src="https://user-images.githubusercontent.com/4212806/191136464-8f3c05fc-9e70-4678-9402-6d4d8232661b.gif" height="25%" width="25%"/><img src="https://user-images.githubusercontent.com/4212806/191136616-06ce3640-7e35-45a3-8b2e-7f7a5b9b7f28.gif" height="25%" width="25%"/><img src="https://user-images.githubusercontent.com/4212806/191136450-4b2d55c1-c3c7-47d6-996e-11c62448747b.gif" height="25%" width="25%"/><img src="https://user-images.githubusercontent.com/4212806/191137124-7dae37a3-a659-4e3e-8373-9a1c44b57e48.gif" height="25%" width="25%"/><img src="https://user-images.githubusercontent.com/4212806/191136896-e42ab4d9-3a2f-4553-a1c7-49c59fc7e7a2.gif" height="25%" width="25%"/>

JetNet is a collection of [models](models.md), [datasets](datasets.md),
[tools](tools.md) that make it easy to explore neural networks on NVIDIA Jetson (and desktop too!). It can easily be used and extended with [Python](python/usage.md).  

Check out the documentation to learn more and get started!

### It's easy to use

JetNet comes with tools that allow you to easily <a href="tools/#build">build</a> , <a href="tools/#profile">profile</a> and <a href="tools/#demo">demo</a> models.  This helps you easily try out models to see what is right for your application.  

```bash
jetnet demo jetnet.trt_pose.RESNET18_HAND_224X224_TRT_FP16
```

<img src="https://user-images.githubusercontent.com/4212806/191137124-7dae37a3-a659-4e3e-8373-9a1c44b57e48.gif"/>


### It's implementation agnostic

JetNet has well defined interfaces for tasks like [classification](python/reference/#classificationmodel), [detection](python/reference/#detectionmodel), [pose estimation](python/reference/#posemodel), and [text detection](python/reference/#textdetectionmodel).  This means models have a familiar interface, regardless of which framework they are implemented in.  As a user, this lets you easily use a variety of models without re-learning
a new interface for each one. 

### It's easy to set up

JetNet comes with pre-built docker containers for Jetson and Desktop.
In case these don't work for you, manual setup instructions are provided.
Check out the <a href="setup">Setup</a> page for details.

### It's extensible

JetNet is written with <a href="python/usage">Python</a> so that it is easy
to extend.  If you want to use the JetNet tools with a different model, or are
considering contributing to the project to help other developers easily use your model, all you need to do is implement one of the JetNet [interfaces](python/reference/#abstract-types) and define a config for
re-creating the model.  


## Get Started!

Head on over the [Documentation](setup) to learn more and get started!
