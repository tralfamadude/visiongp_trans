import logging
import argparse
import ast
import random
import os
import pandas as pd
import xml.etree.ElementTree as ET

# from asn1crypto.cms import ClassList

import vision_genprog.tasks.image_processing as image_processing
import vision_genprog.transPop as transPop
import cv2

#from evolve_image_classification import FilepathClassList

parser = argparse.ArgumentParser()
parser.add_argument('--imagesbeforeDirectory', help="The before images directory. Default: './idata/semiconductor_images/before'", default='./idata/semiconductor_images/before')
parser.add_argument('--imagesafterDirectory', help="The after images directory. Default: './idata/semiconductor_images/after'", default='./idata/semiconductor_images/after')
#parser.add_argument('--classFilename', help="The filename of the classification file. Default: 'class.csv'", default='class.csv')
parser.add_argument('--numberOfIndividuals', help="The number of individuals. Default: 64", type=int, default=64)
parser.add_argument('--levelToFunctionProbabilityDict', help="The probability to generate a function, at each level. Default: '{0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}'", default='{0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}')
parser.add_argument('--proportionOfConstants', help='The probability to generate a constant, when a variable could be used. Default: 0', type=float, default=0)
parser.add_argument('--constantCreationParametersList', help="The parameters to use when creating constants: [minFloat, maxFloat, minInt, maxInt, width, height]. Default: '[-1, 1, 0, 255, 256, 256]'", default='[-1, 1, 0, 255, 256, 256]')
parser.add_argument('--primitivesFilepath', help="The filepath to the XML file for the primitive functions. Default: './vision_genprog/tasks/image_processing.xml'", default='./vision_genprog/tasks/image_processing.xml')
parser.add_argument('--outputDirectory', help="The output directory. Default: './outputs/'", default='./outputs/')
#parser.add_argument('--numberOfGenerations', help="The number of generations to run. Default: 32", type=int, default=32)
parser.add_argument('--minimumValidationAccuracyToStop', help="The minimum validation accuracy to stop the evolution. Default: 0.99", type=float, default=0.99)
parser.add_argument('--weightForNumberOfNodes', help="Penalty term proportional to the number of nodes. Default: 0.001", type=float, default=0.001)
parser.add_argument('--numberOfTournamentParticipants', help="The number of participants in selection tournaments. Default: 2", type=int, default=2)
parser.add_argument('--mutationProbability', help="The probability to mutate a child. Default: 0.2", type=float, default=0.2)
parser.add_argument('--proportionOfNewIndividuals', help="The proportion of randomly generates individuals per generation. Default: 0.2", type=float, default=0.2)
parser.add_argument('--maximumNumberOfMissedCreationTrials', help="The maximum number of missed creation trials. Default: 1000", type=int, default=1000)
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

levelToFunctionProbabilityDict = ast.literal_eval(args.levelToFunctionProbabilityDict)
constantCreationParametersList = ast.literal_eval(args.constantCreationParametersList)
# image_shapeHW = (constantCreationParametersList[5], constantCreationParametersList[4])
image_shapeHW = (620, 623)  # ToDo: get this from an image

def main():
    logging.info("create_transformation_population.py main()")

    # Create the output directory
    if not os.path.exists(args.outputDirectory):
        os.makedirs(args.outputDirectory)

    image_before_filepaths = ImageFilepaths(args.imagesbeforeDirectory)
    #class_df_before = pd.read_csv(args.imagesbeforeDirectory)
    #filepathClass_list_before = os.listdir(args.imagesbeforeDirectory)

    image_after_filepaths = ImageFilepaths(args.imagesafterDirectory)
    pair_image_list = loadImagePairs(image_before_filepaths,image_after_filepaths)


    # Split in train - validation - test
    # Shuffle the list
    random.shuffle(pair_image_list)
    validation_start_ndx1 = round(0.6 * len(pair_image_list))
    test_start_ndx1 = round(0.8 * len(pair_image_list))
    train_filepath_list = pair_image_list[0: validation_start_ndx1]
    validation_filepath_list = pair_image_list[validation_start_ndx1: test_start_ndx1]
    test_filepath_list = pair_image_list[test_start_ndx1:]

    # Create the interpreter
    primitive_functions_tree = ET.parse(args.primitivesFilepath)
    interpreter = image_processing.Interpreter(primitive_functions_tree, image_shapeHW)

    variableName_to_type = {'image': 'grayscale_image'}
    return_type = 'grayscale_image'  # There are two classes

    # Create a population
    trans_pop = transPop.transformationPopulation()
    trans_pop.Generate(
        numberOfIndividuals=args.numberOfIndividuals,
        interpreter=interpreter,
        returnType=return_type,
        levelToFunctionProbabilityDict=levelToFunctionProbabilityDict,
        proportionOfConstants=args.proportionOfConstants,
        constantCreationParametersList=constantCreationParametersList,
        variableNameToTypeDict=variableName_to_type,
        functionNameToWeightDict=None
    )
    # Create the input-output tuples lists
    # train_inputOutputTuples_list_before = InputOutputTuples_before(train_filepathClass_list_before, image_shapeHW)
    # validation_inputOutputTuples_list_before = InputOutputTuples_before(validation_filepathClass_list_before, image_shapeHW)
    # test_inputOutputTuples_list_before = InputOutputTuples_before(test_filepathClass_list_before, image_shapeHW)
    #
    # train_inputOutputTuples_list_after = InputOutputTuples_after(train_filepathClass_list_after, image_shapeHW)
    # validation_inputOutputTuples_list_after = InputOutputTuples_after(validation_filepathClass_list_after, image_shapeHW)
    # test_inputOutputTuples_list_after = InputOutputTuples_after(test_filepathClass_list_after, image_shapeHW)

    # Evaluate the original population
    logging.info("Evaluating the original population...")
    individual_to_cost_dict = trans_pop.EvaluateIndividualCosts(
        inputOutputTuplesList=train_filepath_list,
        variableNameToTypeDict=variableName_to_type,
        interpreter=interpreter,
        returnType=return_type,
        weightForNumberOfElements=args.weightForNumberOfNodes
    )

    logging.info("Starting the population evolution...")
    final_champion = None
    highest_validation_accuracy = 0
    evolution_must_continue = True
    with open(os.path.join(args.outputDirectory, "generations.csv"), 'w+') as generations_file:
        generations_file.write("generation,lowest_cost,median_cost,validation_accuracy\n")
        #for generationNdx in range(1, args.numberOfGenerations + 1):
        generationNdx = 1
        while evolution_must_continue:
            logging.info(" ***** Generation {} *****".format(generationNdx))
            individual_to_cost_dict = trans_pop.NewGenerationWithTournament(
                inputOutputTuplesList=train_filepath_list,
                variableNameToTypeDict=variableName_to_type,
                interpreter=interpreter,
                returnType=return_type,
                numberOfTournamentParticipants=args.numberOfTournamentParticipants,
                mutationProbability=args.mutationProbability,
                currentIndividualToCostDict=individual_to_cost_dict,
                proportionOfConstants=args.proportionOfConstants,
                levelToFunctionProbabilityDict=levelToFunctionProbabilityDict,
                functionNameToWeightDict=None,
                constantCreationParametersList=constantCreationParametersList,
                proportionOfNewIndividuals=args.proportionOfNewIndividuals,
                weightForNumberOfElements=args.weightForNumberOfNodes,
                maximumNumberOfMissedCreationTrials=args.maximumNumberOfMissedCreationTrials
            )
            (champion, lowest_cost) = trans_pop.Champion(individual_to_cost_dict)
            median_cost = trans_pop.MedianCost(individual_to_cost_dict)
            image_path_train = os.path.join(args.outputDirectory, "image_train_output_{}.jpg".format(generationNdx))
            predicted_image_train = interpreter.evaluate(champion)
            cv2.imwrite(predicted_image_train, image_path_train)

            # Validation
            validation_accuracy = transPop.Accuracy(champion, validation_filepath_list, interpreter,
                                                          variableName_to_type,
                                                          return_type)
            logging.info(
                "Generation {}: lowest cost = {}; median cost = {}; validation accuracy = {}".format(generationNdx,
                                                                                                     lowest_cost,
                                                                                                     median_cost,
                                                                                                     validation_accuracy))
            generations_file.write("{},{},{},{}\n".format(generationNdx, lowest_cost, median_cost, validation_accuracy))

            # Save the champion
            champion_filepath = os.path.join(args.outputDirectory,
                                             "champion_{}_{:.4f}_{:.4f}.xml".format(generationNdx, lowest_cost,
                                                                                    validation_accuracy))
            champion.Save(champion_filepath)

            if validation_accuracy > highest_validation_accuracy:
                highest_validation_accuracy = validation_accuracy
                final_champion = champion
            if validation_accuracy >= args.minimumValidationAccuracyToStop:
                evolution_must_continue = False
            generationNdx += 1
        logging.info("Testing the final champion...")
        final_champion_accuracy = transPop.Accuracy(final_champion, test_filepath_list, interpreter,
                                                          variableName_to_type, return_type)
        image_path_test = os.path.join(args.outputDirectory, "image_test_output_{}.jpg".format(generationNdx))
        predicted_image_test = interpreter.evaluate(final_champion)
        cv2.imwrite(predicted_image_test, image_path_test)

        logging.info("final_champion_accuracy = {}".format(final_champion_accuracy))

def ImageFilepaths(images_directory):
    image_filepaths_in_directory = [os.path.join(images_directory, filename) for filename in os.listdir(images_directory)
                              if os.path.isfile(os.path.join(images_directory, filename))
                              and filename.endswith('.jpg')]
    return image_filepaths_in_directory


def loadImagePairs(before_file_list, after_file_list):
    result_list = []  # List[Tuple[Dict[str, Any], Any]]
    for j in range(0,len(before_file_list)):
         before_array = cv2.imread(before_file_list[j], cv2.IMREAD_GRAYSCALE)
         after_array = cv2.imread(after_file_list[j], cv2.IMREAD_GRAYSCALE)
         result_list.append(({'image': before_array}, after_array))
    return result_list


#def FilepathClassList_before(images_directory):
    #filepathClass_list_before = []
    #for index, row in class_df1.iterrows():
        #filename = row['image']
        #classNdx = row['class']
        #print ("filename = {}; classNdx = {}".format(filename, classNdx))
        #filepathClass_list_before.append(images_directory, filename)
    #return filepathClass_list_before

#def FilepathClassList_after(images_directory, class_df2):
    #filepathClass_list_after = []
    #for index, row in class_df2.iterrows():
        #filename = row['image']
        #classNdx = row['class']
        #print ("filename = {}; classNdx = {}".format(filename, classNdx))
        #filepathClass_list_after.append(images_directory, filename)
    #return filepathClass_list_after

#def InputOutputTuples_before(filepathClass_list_before, expected_image_shapeHW, variable_name='image'):
    # List[Tuple[Dict[str, Any], Any]]
    #inputOutput_list_before = []
    #for filepath in filepathClass_list_before:
        #image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        #if image.shape != expected_image_shapeHW:
            #raise ValueError("InputOutputTuples(): The shape of image '{}' ({}) is not the expected shape {}".format(
                #filepath, image.shape, expected_image_shapeHW))
        #inputOutput_list_before.append(({variable_name: image}))
    #return inputOutput_list_before

#def InputOutputTuples_after(filepathClass_list_after, expected_image_shapeHW, variable_name='image'):
    # List[Tuple[Dict[str, Any], Any]]
    #inputOutput_list_after = []
    #for filepath in filepathClass_list_after:
        #image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        #if image.shape != expected_image_shapeHW:
            #raise ValueError("InputOutputTuples(): The shape of image '{}' ({}) is not the expected shape {}".format(
                #filepath, image.shape, expected_image_shapeHW))
        #inputOutput_list_after.append(({variable_name: image}))
    #return inputOutput_list_after


if __name__ == '__main__':
    main()
