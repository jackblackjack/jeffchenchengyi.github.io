---
interact_link: content/machine-learning/09-miscellaneous-topics/cais/beating-google-ocr-with-slab.ipynb
kernel_name: python3
has_widgets: false
title: 'Beating Google's OCR with CAIS++ and SLAB @ USC'
prev_page:
  url: /portfolio/udacity/04-exploring-condos-sg/exploring-house-prices-singapore-part-3-crispdm-non-technical
  title: 'Exploring Condominiums in Singapore'
next_page:
  url: https://github.com/lucashu1/education-deserts
  title: 'Education Deserts'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Beating Google's OCR

*A Center for Artificial Intelligence in Society's Student Branch (CAIS++) project in collaboration with Sol Price School of Public Policy's Spatial Analysis Lab's Dr. Kim and Michael Chiang*

Github Repository: https://github.com/usc-caisplusplus/SLAB

By: Benjamin Brooks and Chengyi (Jeff) Chen



### Abstract

Goal:
- We want to track where particular businesses are operating in LA to better understand the spatial demographic of businesses in LA. For example, where are all the locations in LA that sell liquor / tacos? 

Problem: 
- A regularly-updated dataset of what businesses are currently operating in LA and where these businesses are located at does not exist. We need a dataset like this to track which geographical locations are starting to attract certain businesses. 
- Google Street View Panomaric images (GSV panos) have been obtained and identified as a potential dataset to assist with the goal above.
- Dr. Kim and Michael Chiang have already tried using Google's OCR on the GSV panos, but results have not been fantastic, especially on languages such as Arabic.

Assumptions: 
1. A store will sell items that are in it's store name - e.g. A store named "Del Taco" will sell the food "tacos"

Deadline:
- April 2020



---
## Proposed Solution

```python
def get_locations(word='taco', database=gsv_panos):
    '''
    Function:
    ---------
    Given a word and the database of all Google Street View
    Panoramic images, search through all the GSV Images that 
    contain the word [as a store sign] and return their locations
    
    Parameters:
    -----------
    word: (str) Any word in any language
    database: Database of all Google Street View Panoramic Images in LA
    
    Returns:
    --------
    Locations of where the word has been found through the 
    Google Street View Panoramic Image database
    '''
    
    # 1. Create a dataset of fake GSV panos
    #    that contain the word (Fall 2019)
    fake_gsv_panos = generate_fake_gsv_panos(word='taco', database=gsv_panos)
    
    # 2. Train discriminative model on the 
    #    fake GSV panos (Spring 2019)
    discriminative_model = train_discriminative_model(dataset=fake_gsv_panos)
    
    # 3. Get the locations of where the 
    #    discriminative model predicts to
    #    contain the word (Fall 2019)
    locations = discriminative_model.find(word='taco', database=gsv_panos)
    return locations
```



---
## Spring 2019 (Discriminative Model)

Summary:
- We've built a pipeline that:
    1. Trains on **labelled** GSV panos that contain the word "**liquor**"
    2. Tests on **labelled** GSV panos by classifying panos as either containing the word "**liquor**" or not

Final Pipeline:
1. We used a sliding window algorithm to crop out overlapping sub-images from the panoramic in order to reduce the size of the input we’re working with and saved it in a folder processed_panos, along with the new bounding box coordinates of the sign in the panoramic image (We made sure to not overcrop the panoramic images - “Going over the width / height of the pano”)
2. We trained YOLO to detect the sign “liquor” on all the “positive” images (With the “liquor” sign) from processed_panos | Confidence: ~ 20% on Average
3. Separately, we trained / tested a Binary CNN+Res-net on the “positive” images from processed_panos and “negative” images (sub-images of the panoramic that didn’t contain the sign “liquor”) | Training Accuracy: ~ 95% | Validation Accuracy: ~ 80%
4. We ran YOLO through each sliding window image and for all those that YOLO predicted to have the sign “liquor” above a confidence threshold we set @ 0.4, we save it into a temporary folder, along with the bounding box positions of the signs predicted
5. We run the images from the temporary folder through a Binary CNN + Res-net that we’ve trained which gave us a confidence probability of whether it contained “liquor” or not. The confidence threshold was set @ 0.94
6. Final Metrics: Total Accuracy - 68.0% | Recall: 84.0% | Precision: 63.6% | Sample Size: 100 Panoramic Images (50 positives and 50 negatives)

Other things we tried:
- We tried generating our own synthetic images using gaussian random noise with the word “liquor” and also colluding some of them by randomly placing obstructions in the synthetic images
    1. Using this synthetic dataset we built a really good binary CNN but it did not generalise well to the real panoramic data
    2. We tried using a convolutional auto encoder to explore whether we can “Denoise” the images but that did not work well too (Might have been a technical issue with the training)
    3. Next, we tried to use them as part of the training pipeline to train the Binary CNN + Res-net and Yolo but the results were very bad, possibly because of covariate shift (the synthetic images and real images look no where close)



---
# Fall 2019 (Generative Model)

Summary of what we need to do:
1. Create a generative model that creates fake GSV panos that contain the target word
    - The fake GSV panos should look as similar to the real GSV panos
    - Ideas: 
        1. Maybe we can use a GAN to generate fake GSV panos this time, compared to just using a fake background like last semester when we tried to create the synthetic dataset
        2. Maybe we can find some way to "clean" the actual GSV images that  we need to search through to "simplify" our search, such as removing the unecessary background in GSV panos except for "rectangularish" boxes that might contain signs that will be of interest to us
2. Improve the discriminative model that will be used
    - Get the metrics like total accuracy up
3. Return locations for those panos that the discriminative model identifies to likely contain the target word
4. Use the locations returned to input onto Google Maps and maybe create a UI for the project

