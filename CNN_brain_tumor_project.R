
#####################################################################
### Brain Tumor Classification using Convolutional Neural Network ###
#####################################################################

#source data for this project is obtained from --> https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

###########################
### Setting up for work ###
###########################

library(fs)
library(tensorflow)
library(keras)
library(tfdatasets)
library(zip)
library(imager)
library(ggplot2)
library(gridExtra)
library(reticulate)

setwd("C:/Users/NET02/Desktop/tasks/2024/Artificial_Intteligence/coding/R/CNN_project")

##########################
### Data preprocessing ###
##########################

#The dataset has 4 different brain tumor categories, which are glioma, meningioma, notumor, and pituitary.

#set the directory containing the images
image_dir <- "./brain_tumor_MRI_dataset/Training/glioma"

#Get a list of all image files in the target directory
old_files <- list.files(path = image_dir, pattern = "*.jpg", full.names = TRUE)

#Create new names as replacement for old
new_files <- paste0(image_dir,"/glioma_", 1:length(old_files), ".jpg")

#Now these new files will be copied to the existing folder
file.copy(from = old_files, to = new_files)

#Now that new files with new names have been copied, we no longer need old files. So let's remove them
file.remove(old_files)

#Turning above into function as I am going to repeat this many times
rename_files <- function(image_dir = "./brain_tumor_MRI_dataset/Training", sub_folder_name = "/meningioma", prefix = "/meningioma_"){
  
  image_dir <- paste0(image_dir, sub_folder_name)
  
  old_files <- list.files(path = image_dir, pattern = "*.jpg", full.names = TRUE)
  
  new_files <- paste0(image_dir,prefix, 1:length(old_files), ".jpg")
  
  file.copy(from = old_files, to = new_files)
  
  file.remove(old_files)
}

#now let's change names in other folders using the function just created
#"meningioma" folder
rename_files(image_dir = "./brain_tumor_MRI_dataset/Training", sub_folder_name = "/meningioma", prefix = "/meningioma_")

#"notumor" folder
rename_files(image_dir = "./brain_tumor_MRI_dataset/Training", sub_folder_name = "/notumor", prefix = "/notumor_")

#"pituitary" folder
rename_files(image_dir = "./brain_tumor_MRI_dataset/Training", sub_folder_name = "/pituitary", prefix = "/pituitary_")


#Making the names of testing dataset consistent with training dataset
#"glioma" folder
rename_files(image_dir = "./brain_tumor_MRI_dataset/Testing", sub_folder_name = "/glioma", prefix = "/glioma_")

#"meningioma" folder
rename_files(image_dir = "./brain_tumor_MRI_dataset/Testing", sub_folder_name = "/meningioma", prefix = "/meningioma_")

#"notumor" folder
rename_files(image_dir = "./brain_tumor_MRI_dataset/Testing", sub_folder_name = "/notumor", prefix = "/notumor_")

#"pituitary" folder
rename_files(image_dir = "./brain_tumor_MRI_dataset/Testing", sub_folder_name = "/pituitary", prefix = "/pituitary_")

#Now let's create another folder named brain_tumor_mri_small which I am going to use for the actual CNN model training.  
#under this, I will create three sub-folders - test, train, validation, each of which then will have all appropriate 3 brain tumor categories.

original_dir <- path("./brain_tumor_MRI_dataset/Training") #in order to use relative path, need to type "." before a new path
original_dir <- path("./brain_tumor_MRI_dataset/Testing") #for testing folder file extraction
new_base_dir <- path("./brain_tumor_mri_small")

make_subset <- function(subset_name, start_index, end_index) {
  for (category in c("glioma","meningioma", "notumor", "pituitary")) {
    file_name <- glue::glue("{category}_{start_index:end_index}.jpg")
    dir_create(new_base_dir/subset_name/category)
    file_copy(original_dir/category/file_name,
              new_base_dir/subset_name/category/file_name)
  }
}

make_subset("train", start_index = 1, end_index = 900)
make_subset("validation", start_index = 901, end_index = 1300)
make_subset("test", start_index = 1, end_index = 300) #before running this, need to change original directory above from "Training" "Testing"

# the for loop here iterate through test folders and add prefix "category_test_" to the existing file name
for(category in c("glioma","meningioma", "notumor", "pituitary")){
  name = paste0("/", category)
  prefix = paste0("/",category,"_", "test_")
  rename_files(image_dir = "./brain_tumor_mri_small/test", sub_folder_name = name, prefix = prefix)
}

#in order for us to get the data into the model, we need following steps
#1) Read the picture files.
#2) Decode the JPEG content to RGB grids of pixels.
#3) Convert these into floating-point tensors
#4) Resize them to a shared size (we will use 256 * 256)
#5) Pack them into batches (we'll use batches of 32 images)
# In Keras, image_dataset_from_directory(), lets us quickly take care of above steps. 


train_dataset <- image_dataset_from_directory(new_base_dir/"train",
                                              image_size = c(256,256),
                                              batch_size = 32)


validation_dataset <- image_dataset_from_directory(new_base_dir/"validation",
                                                   image_size = c(256,256),
                                                   batch_size = 32)


test_dataset <- image_dataset_from_directory(new_base_dir/"test",
                                             image_size = c(256,256),
                                             batch_size = 32)



 #Let's look at the output of one of these Dataset objects: it yields batches of 180*180 RGB images(shape(32, 180, 180, 3))
# and integer labels (shape(32)). There are 32 samples in each batch(the batch size)

c(data_batch, labels_batch) %<-% iter_next(as_iterator(train_dataset))
data_batch$shape
labels_batch$shape


#visualize dataset
batch <- train_dataset %>% as_iterator() %>% iter_next()
str(batch)

c(images, labels) %<-% batch

display_image_tensor <- function(x, ..., max = 255, plot_margins = c(0,0,0,0)) {
  
  if(!is.null(plot_margins))
    par(mar=plot_margins)
  
  x %>% 
    as.array() %>%
    drop() %>%
    as.raster(max=max) %>%
    plot(..., interpolate = FALSE)
  
}
par(mfrow = c(3,3))
for (i in 1:9) display_image_tensor(images[i,,,], plot_margins = rep(.5,4))


######################################
### Building the Initial CNN model ###
######################################

inputs <- layer_input(shape=c(256,256,3)) # the model expects RGB images of size 180*180

outputs <- inputs %>%
  layer_rescaling(1 / 255) %>% #Rescale inputs to the [0, 1] range by dividing them by 255
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size =2) %>%
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size =2) %>%
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size =2) %>%
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size =2) %>%
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(4, activation = "softmax")

model <- keras_model(inputs, outputs)
model

# for the compilation step, we will go with the RMSprop optimizer, 
model %>% compile(loss = "sparse_categorical_crossentropy",
                  optimizer = "rmsprop",
                  metrics = c("accuracy"))


#####################
### Fit the model ###
#####################

callbacks <- list(
  callback_model_checkpoint(
    filepath = "convnet_brain_tumor_mri_from_scratch.keras",
    save_best_only = TRUE, 
    monitor = "val_loss"
  )
)


history <- model %>%
  fit(train_dataset,
      epochs = 30,
      validation_data = validation_dataset,
      callbacks = callbacks)



plot(history)

save_model_tf(model, "./convnet_brain_tumor_mri_from_scratch") #saving the entire model

#Evaluating the model on the test set
test_model <- load_model_tf("convnet_brain_tumor_mri_from_scratch.keras")
result <- evaluate(test_model, test_dataset) #initial model achieves 85.08% test accuracy. Seems pretty good!
cat(sprintf("Test accuracy: %.3f\n", result["accuracy"]))

par(mfrow = c(1,1))


display_image_tensor <- function(x, ..., max = 255, plot_margins = c(0,0,0,0)) {
  
  if(!is.null(plot_margins))
    par(mar=plot_margins)
  
  x %>% 
    as.array() %>%
    drop() %>%
    as.raster(max=max) %>%
    plot(..., interpolate = FALSE)
  
}

image_size <- c(256,256)

img_tensor <- 
  #"./brain_tumor_mri_small/test/glioma/glioma_test_100.jpg" %>%
  #"./brain_tumor_mri_small/test/meningioma/meningioma_test_7.jpg" %>%
  "./brain_tumor_mri_small/test/notumor/notumor_test_90.jpg" %>%
  #"./brain_tumor_mri_small/test/pituitary/pituitary_test_3.jpg" %>%
  tf$io$read_file() %>%
  tf$io$decode_image(channels = 3) %>%
  #tf$io$decode_image() %>%
  tf$image$resize(as.integer(image_size)) %>%
  tf$expand_dims(0L)

display_image_tensor(img_tensor)
score <- test_model %>% predict(img_tensor)
sprintf("This image is %.2f%% glioma, %.2f%% meningioma, %.2f%% notumor, %.2f%% pituitary",
        100 * score[1], 100 * score[2], 100 * score[3], 100 * score[4])


### initial model achieves 85% test accuracy. Pretty good!

###############################
### Using data augmentation ###
###############################

data_augmentation <- keras_model_sequential() %>%
  layer_random_flip("horizontal") %>%
  layer_random_rotation(0.1) %>%
  layer_random_zoom(0.2)

#let's look at the augmented images
batch <- train_dataset %>%
  as_iterator() %>%
  iter_next()

c(images, labels) %<-% batch

par(mfrow = c(3,3), mar = rep(.5, 4))


image <- images[6,,,]
plot(as.raster(as.array(image), max=255)) # plot the first image of the batch, without argmentation

for(i in 2:9) {
  augmented_images <- data_augmentation(images)
  augmented_image <- augmented_images[6,,,]
  plot(as.raster(as.array(augmented_image), max=255))
}


#applying data_augmentation
inputs <- layer_input(shape=c(256,256,3)) # the model expects RGB images of size 180*180

outputs <- inputs %>%
  data_augmentation() %>% #including data_augmentation
  layer_rescaling(1 / 255) %>% #Rescale inputs to the [0, 1] range by dividing them by 255
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size =2) %>%
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size =2) %>%
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size =2) %>%
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size =2) %>%
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") %>%
  layer_flatten() %>%
  layer_dropout(0.5) %>% #including layer_dropout layer
  layer_dense(4, activation = "softmax")


model <- keras_model(inputs, outputs)
model
# for the compilation step, we will go with the RMSprop optimizer, 
model %>% compile(loss = "sparse_categorical_crossentropy",
                  optimizer = "rmsprop",
                  metrics = c("accuracy"))

#Let's train the model using data augmentation and dropout. 
callbacks <- list(
  callback_model_checkpoint(
    filepath = "convnet_brain_tumor_mri_with_augmentation_dropout.keras",
    save_best_only = TRUE, 
    monitor = "val_loss"
  )
)


history <- model %>%
  fit(train_dataset,
      epochs = 100,
      validation_data = validation_dataset,
      callbacks = callbacks)

  which.max(history$metrics$val_accuracy)


save_model_tf(model, "./convnet_brain_tumor_mri_with_augmentation") #saving the entire model
  
#Evaluating the model on the test set
test_model <- load_model_tf("convnet_brain_tumor_mri_with_augmentation_dropout.keras")
result <- evaluate(test_model, test_dataset) #with data augmentation, we achived 95.33% test accuracy!
cat(sprintf("Test accuracy: %.3f\n", result["accuracy"]))

par(mfrow = c(1,1))

image_size <- c(256,256)

img_tensor <- 
  #"./brain_tumor_mri_small/test/glioma/glioma_test_280.jpg" %>%
  #"./brain_tumor_mri_small/test/meningioma/meningioma_test_7.jpg" %>%
  #"./brain_tumor_mri_small/test/notumor/notumor_test_120.jpg" %>%
  "./brain_tumor_mri_small/test/pituitary/pituitary_test_6.jpg" %>%
  tf$io$read_file() %>%
  tf$io$decode_image(channels = 3) %>%
  #tf$io$decode_image() %>%
  tf$image$resize(as.integer(image_size)) %>%
  tf$expand_dims(0L)

display_image_tensor(img_tensor)
score <- test_model %>% predict(img_tensor)
sprintf("This image is %.2f%% glioma, %.2f%% meningioma, %.2f%% notumor, %.2f%% pituitary",
        100 * score[1], 100 * score[2], 100 * score[3], 100 * score[4])

#####################################################
### Improving the base model using best practices ###
#####################################################

# this is referring to the chapter 9 of "Deep Learning with R" : residual connection, batch normalization, 
# depthwise separable convolutions. 

# the following model will be built on the following premises
# your model should be organized into repeated blocks of layers, usually made of multiple convolution layers and a max-pooling
# the number of filters in your layers should increases as the size of the spatial feature maps decreases
# Deep and narrow is better than broad and shallow
# Introducing  residual connections around blocks of layers helps you train deeper networks. 
# It can be beneficial to introduce batch normalization layers after your convolution layers.
# It can be beneficial to replace layer_conv_2d() with layer_separable_conv_2d(), which are more parameter efficient. 

data_augmentation <- keras_model_sequential() %>%
  layer_random_flip("horizontal") %>%
  layer_random_rotation(0.1) %>%
  layer_random_zoom(0.2)

inputs <- layer_input(shape=c(256,256,3)) 

x <- inputs %>% 
  data_augmentation() %>%
  layer_rescaling(scale = 1/255)


x <- x %>% layer_conv_2d(32,5, use_bias=FALSE)

for(size in c(32, 64, 128, 256, 512)){
  
  residual <- x  
  
  x <- x %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    
    layer_separable_conv_2d(size, 3, padding = "same", use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    
    layer_separable_conv_2d(size, 3, padding = "same", use_bias = FALSE) %>%
    layer_max_pooling_2d(pool_size = 3, strides = 2, padding = "same")
  
  residual <- residual %>%
    layer_conv_2d(size, 1, strides = 2, padding = "same", use_bias = FALSE)
  
  x <- layer_add(list(x, residual))
}

outputs <- x %>% layer_global_average_pooling_2d() %>% layer_dropout(0.5) %>%
  layer_dense(4, activation = "softmax")

model <- keras_model(inputs, outputs)
model

model %>% compile(loss = "sparse_categorical_crossentropy",
                  optimizer = "rmsprop",
                  metrics = c("accuracy"))

callbacks <- list(
  callback_model_checkpoint(
    filepath = "convnet_brain_tumor_mri_best_practice_model.keras",
    save_best_only = TRUE, 
    monitor = "val_loss"
  )
)


history <- model %>% fit(
  train_dataset,
  epochs = 100, 
  validation_data = validation_dataset,
  callbacks = callbacks
)

which.max(history$metrics$val_accuracy)

save_model_tf(model, "./convnet_brain_tumor_mri_best_practices") #saving the entire model

test_model <- load_model_tf("convnet_brain_tumor_mri_best_practice_model.keras")
result <- evaluate(test_model, test_dataset)
cat(sprintf("Test accuracy: %.3f\n", result["accuracy"]))
#Test accuracy: 0.9700  --> 97%

# Drawing a 2*3 plot grid with predicted/actual labels
load_and_preprocess_image <- function(file, image_size = c(256,256)){
  
  img_tensor <- 
    tf$io$read_file(file) %>%
    tf$io$decode_image(channels = 3) %>%
    tf$image$resize(as.integer(image_size)) %>%
    tf$expand_dims(0L)
  return(img_tensor)
  
}

image_files <- list("./brain_tumor_mri_small/test/glioma/glioma_test_255.jpg",
                    "./brain_tumor_mri_small/test/meningioma/meningioma_test_255.jpg",
                    "./brain_tumor_mri_small/test/notumor/notumor_test_255.jpg", 
                    "./brain_tumor_mri_small/test/pituitary/pituitary_test_255.jpg") 


predict_and_plot <- function(image_files, actual_labels, model, image_size = c(256,256)){
  
  plots <- list()
  
  for(i in seq_along(image_files)){
    img <- load_and_preprocess_image(image_files[[i]][1], image_size)
    pred <- test_model %>% predict(img)
    
    if(which.max(pred) == 1){
     pred_text_label = "Glioma"
    } else if(which.max(pred) == 2){
      pred_text_label = "Meningioma"
      } else if(which.max(pred) == 3){
        pred_text_label = "No_tumor"
      } else {
        pred_text_label = "Pituitary"
        }
    
    pred_label <-paste0(pred_text_label," ",round(pred[which.max(pred)] * 100, 2), "%") 
    
    #read and convert image for plotting
    img_plot <- load.image(image_files[[i]][1])
    img_plot <- as.raster(resize(img_plot, size_x = image_size[1], size_y = image_size[2]))
    
    # Create a ggplot object
    p <- ggplot() +
      annotation_raster(img_plot, xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf) +
      labs(title = paste("Actual: ", actual_labels[i]) , 
           subtitle = paste("Predicted: ", pred_label) 
      )+ 
      theme(plot.title = element_text(size = 12.5, color="blue", face='bold'), 
            plot.subtitle = element_text(size = 12.5, color="red", face='bold') 
      ) 
    
    plots[[i]] <- p
  }
  
  #arrange plots in a grid
  
  grid.arrange(grobs = plots, ncol=2)
  
}


  
actual_labels <- c("Glioma", "Meningioma", "No_tumor", "Pituitary")  

predict_and_plot(image_files, actual_labels, test_model,image_size = c(256,256))

#####################################
### Leveraging a pretrained model ###
#####################################

# There are two ways to use a pretrained model 
# 1) feature extraction and fine-tuning

#Feature extraction with a pretrained model
#feature extraction consists of using the representations learned by a previously trained model to extract interesting features from new samples
#In the case of convnet, feature extraction consists of 
#1)taking the convolutional base of a previously trained network, 
#2)running the new data through it
#3)training a new classifier on top of the output. 

#Instantiating the VGG16 convolutional base

# conv_base <- application_densenet169(
#   weights = "imagenet",
#   include_top = FALSE,
#   input_shape = c(256,256,3)
# )


conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(256,256,3)
)

length(validation_dataset)  


library(listarrays)
#extracting the VGG16 features and corresponding labels
get_features_and_labels <- function(dataset){
  n_batches <- length(dataset)
  all_features <- vector("list", n_batches)
  all_labels <- vector("list", n_batches)
  iterator <- as_array_iterator(dataset)
  for (i in 1:n_batches) {
    c(images, labels) %<-% iter_next(iterator)
    preprocessed_images <- imagenet_preprocess_input(images)
    features <- conv_base %>% predict(preprocessed_images)
    
    all_labels[[i]] <- labels
    all_features[[i]] <- features
  }
  
  all_features <- listarrays::bind_on_rows(all_features)
  all_labels <- listarrays::bind_on_rows(all_labels)
  
  list(all_features, all_labels)
}

c(train_features, train_labels) %<-% get_features_and_labels(train_dataset)
c(val_features, val_labels) %<-% get_features_and_labels(validation_dataset)
c(test_features, test_labels) %<-% get_features_and_labels(test_dataset)

dim(train_features)
dim(val_features)
dim(test_features)
#At this point, we can define our densely connected classifier and train it on the data and labels that we just recorded
inputs <- layer_input(shape = c(8,8,512))

outputs <- inputs %>%
  layer_flatten() %>%
  layer_dense(256) %>%
  layer_dropout(.5) %>%
  layer_dense(4, activation="softmax")

model <- keras_model(inputs, outputs)

# for the compilation step, we will go with the RMSprop optimizer, 
model %>% compile(loss = "sparse_categorical_crossentropy",
                  optimizer = "rmsprop",
                  metrics = c("accuracy"))

#Let's train the model using data augmentation and dropout. 
callbacks <- list(
  callback_model_checkpoint(
    filepath = "convnet_brain_tumor_mri_using_pretrained_model.keras",
    save_best_only = TRUE, 
    monitor = "val_loss"
  )
)


history <- model %>% fit(
  train_features, train_labels,
  epochs = 50, 
  validation_data = list(val_features, val_labels),
  callbacks = callbacks
)


test_model <- load_model_tf("convnet_brain_tumor_mri_using_pretrained_model.keras")
result <- test_model %>% evaluate(test_features, test_labels) #Test accuracy: 0.803
cat(sprintf("Test accuracy: %.3f\n", result["accuracy"]))

par(mfrow = c(1,1))

image_size <- c(256,256)


# Drawing a 2*3 plot grid with predicted/actual labels
load_and_preprocess_image <- function(file, image_size = c(256,256)){
  
  img_tensor <- 
    tf$io$read_file(file) %>%
    tf$io$decode_image(channels = 3) %>%
    tf$image$resize(as.integer(image_size)) %>%
    tf$expand_dims(0L) %>%
    imagenet_preprocess_input()
  return(img_tensor)
  
}

image_files <- list("./brain_tumor_mri_small/test/glioma/glioma_test_120.jpg",
                    "./brain_tumor_mri_small/test/meningioma/meningioma_test_120.jpg",
                    "./brain_tumor_mri_small/test/notumor/notumor_test_120.jpg", 
                    "./brain_tumor_mri_small/test/pituitary/pituitary_test_120.jpg") 


predict_and_plot <- function(image_files, actual_labels, model, image_size = c(256,256)){
  
  plots <- list()
  
  for(i in seq_along(image_files)){
    
    
    img <- load_and_preprocess_image(image_files[[i]][1], image_size)
    feature <- conv_base %>% predict(img)
    pred <- test_model %>% predict(feature)
    
    if(which.max(pred) == 1){
      pred_text_label = "Glioma"
    } else if(which.max(pred) == 2){
      pred_text_label = "Meningioma"
    } else if(which.max(pred) == 3){
      pred_text_label = "No_tumor"
    } else {
      pred_text_label = "Pituitary"
    }
    
    pred_label <-paste0(pred_text_label," ",round(pred[which.max(pred)] * 100, 2), "%") 
    
    #read and convert image for plotting
    img_plot <- load.image(image_files[[i]][1])
    img_plot <- as.raster(resize(img_plot, size_x = image_size[1], size_y = image_size[2]))
    
    # Create a ggplot object
    p <- ggplot() +
      annotation_raster(img_plot, xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf) +
      labs(title = paste("Actual: ", actual_labels[i]) , 
           subtitle = paste("Predicted: ", pred_label) 
      )+ 
      theme(plot.title = element_text(size = 12.5, color="blue", face='bold'), 
            plot.subtitle = element_text(size = 12.5, color="red", face='bold') 
      ) 
    
    plots[[i]] <- p
  }
  
  #arrange plots in a grid
  
  grid.arrange(grobs = plots, ncol=2)
  
}

actual_labels <- c("Glioma", "Meningioma", "No_tumor", "Pituitary")  
predict_and_plot(image_files, actual_labels, test_model,image_size = c(256,256))

#As can be seen using pretrained model may not boost performance, and there are several reasons for that according to Chat GPT..
# Using pretrained models such as ResNet and VGG16 typically provides a boost in performance due to the wealth of knowledge they contain from being trained on large-scale datasets like ImageNet. However, there are several reasons why you might observe a drop in performance when switching from a custom model to a pretrained model:
# 
# 1. **Mismatch in Domain**: Pretrained models like ResNet and VGG16 are initially trained on ImageNet, which consists of everyday images, objects, and scenes. If your MRI brain tumor dataset significantly differs in terms of features, scales, textures, and patterns, the pretrained weights might not be as useful. Medical images, particularly MRIs, have characteristics that differ greatly from natural images.
# 
# 2. **Fine-tuning Strategy**: The method of fine-tuning the pretrained model can significantly impact its performance. If you do not fine-tune the pretrained models correctly (for example, not allowing enough layers to be trainable or not setting appropriate learning rates), the model may not adapt well to the new dataset.
#    - **Frozen Layers**: If too many layers are frozen, the model might not learn the specific features of your MRI dataset.
#    - **Learning Rate**: Using the wrong learning rate can prevent the model from converging properly. Fine-tuning might require a different learning rate schedule from training a model from scratch.
# 
# 3. **Data Preprocessing**: Differences in data preprocessing methods between your initial custom CNN model and the pretrained models could affect the performance. The pretrained models might expect images to be normalized in a specific way (for instance, with mean and standard deviation based on the ImageNet dataset).
# 
# 4. **Model Complexity**: Your custom model might be more optimized for the specific features of your dataset, containing architectural choices like residual connections and batch normalization, which are tailored to work well with your images.
# 
# 5. **Training Epochs and Batch Size**: Differences in the number of training epochs, batch size, and other training hyperparameters could also impact the model's performance.
# 
# 6. **Class Imbalance**: If there is an imbalance in the distribution of your classes, the pretrained models might not handle this as effectively as your custom model, especially if additional techniques like class weighting or balanced sampling were used with your custom model.
# 
# 7. **Overfitting**: Your custom model might be overfitting to your specific dataset, capturing nuances that do not generalize well. The pretrained models could be regularizing too much, failing to capture these nuances, or vice versa.
# 
# ### Steps to Improve Performance with Pretrained Models:
# 
# 1. **Layer Freezing and Unfreezing**: Experiment with different numbers of frozen layers and ensure that the right amount of the model is trainable to adapt to your specific task. Typically, you start by freezing most of the layers and then gradually unfreeze more as training proceeds.
#    
# 2. **Learning Rate Tuning**: Fine-tune the learning rate. Often, a lower learning rate is used when fine-tuning pretrained networks.
# 
# 3. **Data Augmentation**: Ensure that your preprocessing and data augmentation steps match the expectations of the pretrained models. Use similar augmentation strategies that worked for your initial model.
#    
# 4. **Normalization**: Preprocess the images in a way that matches what the pretrained model expects. For example, normalizing the pixel values using the mean and standard deviation of the ImageNet dataset if using ImageNet pretrained weights.
# 
# 5. **Model Architecture**: Consider adding custom layers on top of the pretrained model to better capture the features specific to your MRI dataset.


# So I may need to try fine-tuning using "Google Colab"  #

##########################################
### Fine-tuning using pretrained model ###
##########################################

#fine-tuning was performed in Google Colab using Python, 
# The following elaborates the process of fine tuning... this is important (from p234 of Deep learning with r)
# 1. Add our custom network on top of an already-trained base network
# 2. Freeze the base network
# 3. Train the part we added (i.e., the top classifier)
# 4. Unfreeze some layers in the base network. (last three layer in this case, which correspond to more abstract feature) 
# 5. Jointly train both these layers and the part we added (caution.... remember to adjust to a very low learning rate = 0.000001)


#here is the python code used for fine-tuning

## Feature Extraction with Data Augmentation
conv_base = keras.applications.vgg16.VGG16(
  weights="imagenet",
  include_top=False)
conv_base.trainable = False


data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
  ]
)

inputs = keras.Input(shape=(256, 256, 3))
x = data_augmentation(inputs)
x = keras.applications.vgg16.preprocess_input(x)
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(4, activation="softmax")(x)
model = keras.Model(inputs, outputs)


model.compile(loss="sparse_categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

callbacks = [
  keras.callbacks.ModelCheckpoint(
    filepath="feature_extraction_with_data_augmentation.keras",
    save_best_only=True,
    monitor="val_loss")
]


history = model.fit(
  train_dataset,
  epochs=50,
  validation_data=validation_dataset)
#callbacks=callbacks)

# Now that the top classifier has been trained, we are going to unfreeze several layers
conv_base.trainable = True
for layer in conv_base.layers[:-4]:
  layer.trainable = False


# we are going to use very low learning rate.
# The reason for using a low learning rate is that we want to limit the magnitude of the modifications we make to the representations of the three layers we are fine tuning. 
# Updates that are too large may harm these representations. 
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),   # we are going to use very low learning rate
              metrics=["accuracy"])

callbacks = [keras.callbacks.ModelCheckpoint(
  filepath="fine_tuning.keras",
  save_best_only=True,
  monitor="val_loss")
]

history = model.fit(
  train_dataset,
  epochs=30,
  validation_data=validation_dataset)
#callbacks=callbacks)

test_loss, test_accuracy = model.evaluate(test_dataset)

#38/38 [==============================] - 5s 127ms/step - loss: 6.4851 - accuracy: 0.9400
#now it achieves up to 94% accuracy.... compared to "feature extraction with data augmentation" this is about 4% improvement. Great!

#the following shows how to import the model created in python(the final model was exported from google colab) 

library(reticulate)

#how to find python interpreter
py_path <- py_discover_config("python")$python
print(py_path)

use_python("C:/Users/NET02/Documents/.virtualenvs/r-tensorflow/Scripts/python.exe")

#Bringing the find-tuned model and evaluating the model on the test set
test_model_fine_tuned <- tf$keras$models$load_model("brain_tumor_fined_tuned")
result <- evaluate(test_model_fine_tuned, test_dataset) #fine tuning with vgg16 achieved 94% test accuracy!
cat(sprintf("Test accuracy: %.3f\n", result["accuracy"]))

# brining the result from DenseNet169 this model is also run in google colab in python//.
test_model_fine_tuned <- tf$keras$models$load_model("brain_tumor_fined_tuned_densenet169")
result <- evaluate(test_model_fine_tuned, test_dataset) #fine tuning with densenet169 achieves 94.67% test accuracy! 
cat(sprintf("Test accuracy: %.3f\n", result["accuracy"]))


test_model_fine_tuned <- tf$keras$models$load_model("brain_tumor_fined_tuned_densenet169_further_adjusted")
result <- evaluate(test_model_fine_tuned, test_dataset) 
#fine tuning with densenet169 (batch_normalization, regularization) achieves 95.50% test accuracy! 
cat(sprintf("Test accuracy: %.3f\n", result["accuracy"]))



# Drawing a 2*3 plot grid with predicted/actual labels
load_and_preprocess_image <- function(file, image_size = c(256,256)){
  
  img_tensor <- 
    tf$io$read_file(file) %>%
    tf$io$decode_image(channels = 3) %>%
    tf$image$resize(as.integer(image_size)) %>%
    tf$expand_dims(0L)
  return(img_tensor)
  
}

image_files <- list("./brain_tumor_mri_small/test/glioma/glioma_test_1.jpg",
                    "./brain_tumor_mri_small/test/meningioma/meningioma_test_2.jpg",
                    "./brain_tumor_mri_small/test/notumor/notumor_test_3.jpg", 
                    "./brain_tumor_mri_small/test/pituitary/pituitary_test_4.jpg") 


predict_and_plot <- function(image_files, actual_labels, model, image_size = c(256,256)){
  
  plots <- list()
  
  for(i in seq_along(image_files)){
    img <- load_and_preprocess_image(image_files[[i]][1], image_size)
    pred <- test_model %>% predict(img)
    
    if(which.max(pred) == 1){
      pred_text_label = "Glioma"
    } else if(which.max(pred) == 2){
      pred_text_label = "Meningioma"
    } else if(which.max(pred) == 3){
      pred_text_label = "No_tumor"
    } else {
      pred_text_label = "Pituitary"
    }
    
    pred_label <-paste0(pred_text_label," ",round(pred[which.max(pred)] * 100, 2), "%") 
    
    #read and convert image for plotting
    img_plot <- load.image(image_files[[i]][1])
    img_plot <- as.raster(resize(img_plot, size_x = image_size[1], size_y = image_size[2]))
    
    # Create a ggplot object
    p <- ggplot() +
      annotation_raster(img_plot, xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf) +
      labs(title = paste("Actual: ", actual_labels[i]) , 
           subtitle = paste("Predicted: ", pred_label) 
      )+ 
      theme(plot.title = element_text(size = 12.5, color="blue", face='bold'), 
            plot.subtitle = element_text(size = 12.5, color="red", face='bold') 
      ) 
    
    plots[[i]] <- p
  }
  
  #arrange plots in a grid
  
  grid.arrange(grobs = plots, ncol=2)
  
}

actual_labels <- c("Glioma", "Meningioma", "No_tumor", "Pituitary")  
predict_and_plot(image_files, actual_labels, test_model_fine_tuned,image_size = c(256,256))



############################
### Explaining CNN model ###
############################

model <- load_model_tf("convnet_brain_tumor_mri_best_practice_model.keras")
model


library(tfdatasets)

tf_read_image <-
  function(path, format = "image", resize = NULL, ...) {
    
    img <- path %>%
      tf$io$read_file() %>%
      tf$io[[paste0("decode_", format)]](...)
    
    if (!is.null(resize))
      img <- img %>%
        tf$image$resize(as.integer(resize))
    
    img
  }


# Read the image as RGB instead of grayscale
file_path <- "./brain_tumor_mri_small/test/glioma/glioma_test_127.jpg"
img_raw <- tf$io$read_file(file_path)
img_tensor_rgb <- tf$image$decode_image(img_raw, channels = 3)

# Define the resized image size as a 1-D int32 Tensor
resize_shape <- tf$constant(c(256L, 256L), dtype = tf$int32)

# Resize the image tensor
img_tensor_resized <- tf$image$resize(img_tensor_rgb, size = resize_shape)

# Display the image tensor
display_image_tensor(img_tensor_resized)


conv_layer_s3_classname <- class(layer_conv_2d(NULL, 1, 1))[1]
pooling_layer_s3_classname <- class(layer_max_pooling_2d(NULL))[1]

is_conv_layer <- function(x) inherits(x, conv_layer_s3_classname)
is_pooling_layer <- function(x) inherits(x, pooling_layer_s3_classname)

layer_outputs <- list()
for (layer in model$layers)
  if (is_conv_layer(layer) || is_pooling_layer(layer))
    layer_outputs[[layer$name]] <- layer$output

activation_model <- keras_model(inputs = model$input,
                                outputs = layer_outputs)

activations <- activation_model %>%
  predict(img_tensor_resized[tf$newaxis, , , ])

str(activations)

first_layer_activation <- activations[[ names(layer_outputs)[1] ]]
dim(first_layer_activation)

plot_activations <- function(x, ...) {
  
  x <- as.array(x)
  
  if(sum(x) == 0)
    return(plot(as.raster("gray")))
  
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(x), asp = 1, axes = FALSE, useRaster = TRUE,
        col = terrain.colors(256), ...)
}

plot_activations(first_layer_activation[, , , 5])


activations <- activation_model %>%
  predict(img_tensor_resized[tf$newaxis, , , ])


str(activations)


first_layer_activation <- activations[[ names(layer_outputs)[1] ]]

dim(first_layer_activation)


plot_activations <- function(x, ...) {
  x <- as.array(x)
  if(sum(x) == 0)
    return(plot(as.raster("gray")))
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(x), asp = 1, axes = FALSE, useRaster = TRUE,
        col = terrain.colors(256), ...)
}

plot_activations(first_layer_activation[, , , 7])

for (layer_name in names(layer_outputs)) {
  layer_output <- activations[[layer_name]]
  n_features <- dim(layer_output) %>% tail(1)
  par(mfrow = n2mfrow(n_features, asp = 1.75),
      mar = rep(.1, 4), oma = c(0, 0, 1.5, 0))
  for (j in 1:n_features)
    plot_activations(layer_output[, , , j])
  title(main = layer_name, outer = TRUE)
}


################################################
### Visualizing Heatmaps of class activation ###
################################################

model <- load_model_tf("convnet_brain_tumor_mri_best_practice_model.keras")
model

par(mfrow = c(1,1))

image_size <- c(256,256)

img_tensor <- 
  #"./brain_tumor_mri_small/test/glioma/glioma_test_256.jpg" %>%
  "./brain_tumor_mri_small/test/meningioma/meningioma_test_170.jpg" %>%
  #"./brain_tumor_mri_small/test/notumor/notumor_test_90.jpg" %>%
  #"./brain_tumor_mri_small/test/pituitary/pituitary_test_29.jpg" %>%
  tf$io$read_file() %>%
  tf$io$decode_image(channels = 3) %>%
  #tf$io$decode_image() %>%
  tf$image$resize(as.integer(image_size)) %>%
  tf$expand_dims(0L)

file_path <-   "./brain_tumor_mri_small/test/meningioma/meningioma_test_170.jpg"

display_image_tensor(img_tensor)
score <- model %>% predict(img_tensor)
sprintf("This image is %.2f%% glioma, %.2f%% meningioma, %.2f%% notumor, %.2f%% pituitary",
        100 * score[1], 100 * score[2], 100 * score[3], 100 * score[4])


#First, we create a model that maps that input image to the activations of the last convolutional layer
#setting up a model that returns the last convolutional output
last_conv_layer_name <- "conv2d_15"
classifier_layer_names <- c("global_average_pooling2d","dense_2")
last_conv_layer <- model %>% get_layer(last_conv_layer_name)
last_conv_layer_model <- keras_model(model$inputs, last_conv_layer$output)

#second, we create a model that maps the activations of the last convolutional layer to the final class predictions. 
classifier_input <- layer_input(batch_shape = last_conv_layer$output$shape)

x <- classifier_input
for (layer_name in classifier_layer_names)
  x <- get_layer(model, layer_name)(x)

classifier_model <- keras_model(classifier_input,x)

#Then we compute the gradient of the top predicted class for our input image with respect to the activations of the last convolution layer
with (tf$GradientTape() %as% tape, {
  last_conv_layer_output <- last_conv_layer_model(img_tensor) #preprocessed_img
  tape$watch(last_conv_layer_output)
  preds <- classifier_model(last_conv_layer_output)
  top_pred_index <- tf$argmax(preds[1, ])
  top_class_channel <- preds[, top_pred_index, style = "python"]
})

grads <- tape$gradient(top_class_channel, last_conv_layer_output)

pooled_grads <- mean(grads, axis = c(1, 2, 3), keepdims = TRUE)


heatmap <-
  (last_conv_layer_output * pooled_grads) %>%
  mean(axis = -1) %>%
  .[1, , ]

par(mar=c(0,0,0,0))
plot_activations(heatmap)


#superimposing the heatmap on the original picture

pal <- hcl.colors(256, palette = "RdYlBu", alpha = .75, rev = TRUE)
heatmap <- as.array(heatmap)
heatmap[] <- pal[cut(heatmap, 256)]
heatmap <- as.raster(heatmap)
img <- tf_read_image(file_path, resize = NULL)
display_image_tensor(img)
rasterImage(heatmap, 0, 0, ncol(img), nrow(img), interpolate = TRUE)

