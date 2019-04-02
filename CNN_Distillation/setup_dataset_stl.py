from utils.stl import download_and_extract,read_all_images,read_labels,save_images


# download data if needed
download_and_extract()

# test to check if the whole dataset is read correctly
images = read_all_images(DATA_PATH)
print(images.shape)

labels = read_labels(LABEL_PATH)
print(labels.shape)

# save images to disk
save_images(images, labels , "train")

# test to check if the whole dataset is read correctly
images = read_all_images(TEST_DATA_PATH)
print(images.shape)

labels = read_labels(TEST_LABEL_PATH)
print(labels.shape)

# save images to disk
save_images(images, labels , "val")