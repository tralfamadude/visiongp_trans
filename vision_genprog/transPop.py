import genprog.evolution as gpevo
import genprog
from typing import Dict, List, Any, Set, Optional, Union, Tuple
import numpy as np
import cv2


class transformationPopulation(gpevo.Population):
    def __init__(self):
        pass

    #  evaluate
    def EvaluateIndividualCosts(self, inputOutputTuplesList: List[ Tuple[ Dict[str, Any], Any ] ],
                                variableNameToTypeDict: Dict[str, str],
                                interpreter: genprog.core.Interpreter,
                                returnType: str,
                                weightForNumberOfElements: float) -> Dict[genprog.core.Individual, float]:
        individual_to_cost = {}
        if len(inputOutputTuplesList) == 0:
            raise ValueError("transPop.transformationPopulation.EvaluateIndividualCosts(): len(inputOutputTuplesList) == 0")
        for individual in self._individualsList:
            cost_sum = 0.0
            for inputOutput in inputOutputTuplesList:
                variableName_to_value = inputOutput[0]  # dict
                target_image = inputOutput[1]
                predicted_image = interpreter.Evaluate(individual, variableNameToTypeDict,
                                                             variableName_to_value, returnType)
                fraction_different = image_compare(target_image, predicted_image)
                cost_sum += fraction_different
            # calculate average over test cases.
            individual_to_cost[individual] = cost_sum / len(inputOutputTuplesList)

        if weightForNumberOfElements != 0:
            for individual in self._individualsList:
                individual_to_cost[individual] += weightForNumberOfElements * len(individual.Elements())

        return individual_to_cost


def InputToPrediction(individual, variableNameToValue_list, interpreter, variableName_to_type,
                      return_type):
    correspondingPrediction_list = []
    for variableName_to_value in variableNameToValue_list:
        predicted_class_vector = interpreter.Evaluate(
            individual,
            variableName_to_type,
            variableName_to_value,
            return_type)
        correspondingPrediction_list.append(predicted_class_vector)
    return correspondingPrediction_list

def Accuracy(individual, inputOutput_list, interpreter, variableName_to_type,
                      return_type, save_predicted_images=False, output_dir: str = ".", kind: str = "",
             stop_threshold: int = 0.99):
    if len(inputOutput_list) == 0:
        raise ValueError("transPop.Accuracy(): Empty input-output list")
    correspondingPredictions_list = InputToPrediction(individual, [input for (input, output) in inputOutput_list],
                                            interpreter, variableName_to_type, return_type)
    number_of_correct_predictions = 0
    cost_sum = 0.0
    for sampleNdx in range(len(inputOutput_list)):
        # before image # variableName_to_value = inputOutput_list[sampleNdx][0]
        target_image = inputOutput_list[sampleNdx][1]
        predicted_image = correspondingPredictions_list[sampleNdx]
        fraction_different = image_compare(target_image, predicted_image)
        cost_sum += fraction_different
        if fraction_different < (1.0 - stop_threshold):
            number_of_correct_predictions += 1
        print("{}: fdiff={:.4f}".format(kind, fraction_different))
        if save_predicted_images:
            # save the image in outputs directory
            cv2.imwrite(f"{output_dir}/predicted_{kind}_{sampleNdx}.jpg", predicted_image)
            cv2.imwrite(f"{output_dir}/diff_{kind}_{sampleNdx}.jpg", np.abs(target_image - predicted_image))
    return number_of_correct_predictions/len(inputOutput_list)

#  returns value 0.0 (identical) to 1.0 (totally different)
def image_compare(image1, image2):
    diff_flat = np.ndarray.flatten(np.abs(image1 - image2))
    diff_count = sum(diff_flat)
    fraction_different = diff_count / (len(diff_flat) * 255)
    return fraction_different