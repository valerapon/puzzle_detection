import numpy as np

import cv2

from PIL import Image, ImageOps, ImageFont, ImageDraw

import argparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser(
        description='Detective puzzle on image')
    parser.add_argument('--image', type=str,
                        help='path to content image', required=True)
    return parser.parse_args()  


def image_binarization(image):
    blurred  = cv2.pyrMeanShiftFiltering(image, 31, 91) 
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return threshold


def get_puzzle_binarization(puzzle):
    gray = cv2.cvtColor(puzzle, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    max_con_len = contours[0].shape[0]
    max_con = contours[0]
    for con in contours:
        if con.shape[0] > max_con_len:
            max_con = con
            max_con_len = con.shape[0]
    main_con = [max_con]

    output = cv2.fillPoly(puzzle * 0 + 255, main_con, (0, 0, 0))
    
    max_d = max(output.shape) + 100
    bin_img = np.zeros((max_d, max_d, 3), dtype='uint8') + 255
    bin_img[(max_d - output.shape[0]) // 2 :(max_d - output.shape[0]) // 2  + output.shape[0], \
            (max_d - output.shape[1]) // 2 : (max_d - output.shape[1]) // 2 + output.shape[1], :] = output
    bin_img[bin_img == 0] = 1
    bin_img[bin_img == 255] = 0
    bin_img[bin_img == 1] = 255
    return bin_img


def find_threshold(contours):
    areas, perimetrs = [], []
    for con in contours:
        areas.append(cv2.contourArea(con))
        perimetrs.append(cv2.arcLength(con, closed=True))
    threshold_ar = np.mean(sorted(areas, reverse=True)[:100])
    threshold_pr = np.mean(sorted(perimetrs, reverse=True)[:100])
    return threshold_ar, threshold_pr


def filter_contours(contours, threshold_ar, threshold_pr):
    filter_contours = []
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > threshold_ar and cv2.arcLength(contours[i], closed=True) > threshold_pr:
            filter_contours.append(contours[i])
    return filter_contours


def find_puzzles(image, contours):   
    puzzles = []
    for c in contours:
        figure = c[:,0,:]
        B, A = np.min(figure, axis=0)
        D, C = np.max(figure, axis=0)
        puzzles.append(image[max(0, A - 10): min(image.shape[0] - 1, C + 10), max(0, B - 10): min(image.shape[1] -1, D + 10), :])
    return puzzles


def get_test_sample(puzzles):
    puzzle_image = []
    for puz in puzzles:
        puzzle_image.append(Image.fromarray(get_puzzle_binarization(puz)).resize((30, 30)))

    X_test = np.zeros((len(puzzle_image), 900))
    for i, puz in enumerate(puzzle_image):
        X_test[i] = np.array(puz)[:, :, 0].ravel()
    return X_test


def predict_puzzles_form(model, X_test):
    return model.predict(X_test)


def draw_object_on_image(image, contours, ans_1, ans_2):
    output = Image.fromarray(image)
    draw = ImageDraw.Draw(output)
    font = ImageFont.truetype("arial.ttf", 100)
    for i, c in enumerate(contours):
        figure = c[:, 0, :]
        B, A = np.min(figure, axis=0)
        D, C = np.max(figure, axis=0)
        draw.rectangle([(B, A), (D, C)], outline ='green', width=10)
        
    for i, c in enumerate(contours):
        draw.text(np.min(c[:, 0, :], axis=0), 'P' + str(ans_1[i]) + 'B' + str(ans_2[i]), font=font, fill='blue')
    return output


def main(args):
    image = np.array(Image.open(args.image))
    threshold = image_binarization(image)
    contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    threshold_ar, threshold_pr = find_threshold(contours)
    contours = filter_contours(contours, threshold_ar, threshold_pr)
    puzzles = find_puzzles(image, contours)

    X_test = get_test_sample(puzzles)
    ans_1 = predict_puzzles_form(pickle.load(open('model_1.sav', 'rb')), X_test)
    ans_2 = predict_puzzles_form(pickle.load(open('model_2.sav', 'rb')), X_test)

    output = draw_object_on_image(image, contours, ans_1, ans_2)
    output.save('output/output.png')

    f = open('output/ans.txt', 'w')
    f.write('Всего пазлов: ' + str(len(puzzles)) + '\n')
    f.write('Коды: ' + ' '.join(['P' + str(ans_1[i]) + 'B' + str(ans_2[i]) for i, c in enumerate(contours) ]))
    f.close()
    

if __name__ == '__main__':
    args = parse_args()
    main(args)