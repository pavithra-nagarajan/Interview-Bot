from flask import Flask, jsonify
import mysql.connector
from urllib.parse import urlparse
import os
import whisper
import spacy_universal_sentence_encoder
import torch
import requests
import tempfile
import statistics
from datetime import datetime
import boto3

s3_client = boto3.client("s3")


class AIServerProcessingHistory:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class ImInternQuestionAnswer:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class ImInternAssessmentQuestion:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class QuestionAnswer:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Question:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


app = Flask(__name__)


@app.route("/interview-bot/ai-analysis", methods=["PUT"])
def process_audio_files_using_ai():
    # parse the URL using urlparse function
    master_url = urlparse(os.environ["MASTER_JDBC_URL"])
    replica_url = urlparse(os.environ["REPLICA_JDBC_URL"])
    bucket_name = os.environ["AWS_RESOURCE_BUCKET"]

    # establish a read connection to the database
    read_connection = mysql.connector.connect(
        host=replica_url.path.split("://", 1)[-1].split("/", 1)[0].split(":")[0],
        user=os.environ["REPLICA_JDBC_USERNAME"],
        password=os.environ["REPLICA_JDBC_PASSWORD"],
        database=replica_url.path.split("/")[-1],
    )

    # establish a write connection to the database
    write_connection = mysql.connector.connect(
        host=master_url.path.split("://", 1)[-1].split("/", 1)[0].split(":")[0],
        user=os.environ["MASTER_JDBC_USERNAME"],
        password=os.environ["MASTER_JDBC_PASSWORD"],
        database=master_url.path.split("/")[-1],
    )

    # create a read & write cursor objects to interact with the database
    read_cursor = read_connection.cursor()
    write_cursor = write_connection.cursor()

    # fetching ai_server_processing_history for unprocessed files
    read_cursor.execute(
        "SELECT * FROM ai_server_processing_history where is_processed is false"
    )
    processing_histories = read_cursor.fetchall()

    # checking whether there are any files to process at all
    if check_list_is_valid(processing_histories):
        ai_server_processing_histories = []
        ai_server_processing_history_ids = []

        # process the fetched ai_server_processing_history from db
        for row in processing_histories:
            # dynamically create a dictionary of column names and values for this row
            column_names = [column[0] for column in read_cursor.description]
            row_dict = dict(zip(column_names, row))

            # create an instance of the Processing History object using the dictionary
            processing_history = AIServerProcessingHistory(**row_dict)
            ai_server_processing_histories.append(processing_history)
            ai_server_processing_history_ids.append(processing_history.IM_HISTORY_ID)

        # Construct a comma-separated string of Processing History IDs
        processing_history_ids_str = ",".join(
            str(id) for id in ai_server_processing_history_ids
        )

        # fetching im_intern_coding_assessment_question for unprocessed history ids
        read_cursor.execute(
            "SELECT icaq.* FROM im_intern_coding_assessment_question icaq LEFT JOIN im_question_type iqt ON iqt.ID = icaq.QUESTION_TYPE_ID WHERE iqt.TYPE = 'Video Interview' AND icaq.INTERN_CODING_ASSMT_HISTORY_ID IN ({})".format(
                processing_history_ids_str
            )
        )
        im_questions = read_cursor.fetchall()

        # checking whether there are any questions present for processing
        if check_list_is_valid(im_questions):
            im_intern_coding_assessment_questions = []
            # prepared update query with InProgress and Complete flags in ai_server_processing_history table
            update_in_progress_status_query = "UPDATE ai_server_processing_history SET STATUS = 'InProgress' , STARTED_ON = UTC_TIMESTAMP() WHERE ID = %s"
            update_complete_status_query = "UPDATE ai_server_processing_history SET IS_PROCESSED = TRUE , STATUS = 'Complete' , COMPLETED_ON = UTC_TIMESTAMP() WHERE ID = %s"
            # prepared update query for im_question_answer & processing_job_history once openAI and NLP process is done
            update_questions_answer_query = "UPDATE im_question_answer SET CANDIDATE_ANSWER = %s, AI_SCORE = %s , MODIFIED_ON = UTC_TIMESTAMP() WHERE ID = %s"

            # mapping the cursor values to ImInternAssessmentQuestion class
            im_intern_coding_assessment_questions = process_rows(
                read_cursor, im_questions, class_type=ImInternAssessmentQuestion
            )

            # iterating the ai server history table entries
            for ai_processing_history in ai_server_processing_histories:
                # updating the status flag to indicate the server is under processing state
                write_cursor.execute(
                    update_in_progress_status_query, (ai_processing_history.ID,)
                )
                write_connection.commit()

                # filter im_intern_coding_assessment_questions based on history_id using streams
                read_cursor.execute(
                    "SELECT icah.IM_TEST_INVITATION_ID FROM intern_coding_assmt_history icah WHERE icah.id = %s",
                    (ai_processing_history.IM_HISTORY_ID,),
                )
                im_test_invitation_id = read_cursor.fetchone()

                filtered_im_questions = list(
                    filter(
                        lambda im_question, history_id=ai_processing_history.IM_HISTORY_ID: im_question.INTERN_CODING_ASSMT_HISTORY_ID
                        == history_id,
                        im_intern_coding_assessment_questions,
                    )
                )

                if check_list_is_valid(filtered_im_questions):
                    question_id_map = {}
                    im_question_answer_list = []

                    # mapping the cursor results to ImInternAssessmentQuestion class
                    filtered_im_questions_mapped = [
                        ImInternAssessmentQuestion(**question.__dict__)
                        for question in filtered_im_questions
                    ]

                    # creating a map of Question Id and Imocha Question Id from im_intern_coding_assessment_question
                    im_assmt_question_map = dict(
                        (im_assmt_question.ID, im_assmt_question.QUESTION_ID)
                        for im_assmt_question in filtered_im_questions_mapped
                    )

                    # extract the ID column values from filtered_im_questions and store them as a new list using streams
                    filtered_question_ids = list(
                        map(
                            lambda im_assmt_question: im_assmt_question.ID,
                            filtered_im_questions_mapped,
                        )
                    )

                    # Construct a comma-separated string of running ids
                    filtered_question_ids_str = ",".join(
                        str(id) for id in filtered_question_ids
                    )

                    # fetching the unprocessed candidate answer data for AI analysis
                    read_cursor.execute(
                        "SELECT * FROM im_question_answer iqa WHERE iqa.AI_SCORE IS NULL AND IM_VIDEO_ANSWER_URL IS NOT NULL AND iqa.INTERN_ASSMT_QUESTION_ID IN ({})".format(
                            filtered_question_ids_str
                        )
                    )
                    im_question_answers = read_cursor.fetchall()

                    if check_list_is_valid(im_question_answers):
                        qn_answers_list = []

                        # extract the question ID column values from filtered_im_questions and store them as a new list using streams
                        filtered_im_question_ids = list(
                            map(
                                lambda im_assmt_question: im_assmt_question.QUESTION_ID,
                                filtered_im_questions_mapped,
                            )
                        )

                        # Construct a comma-separated string of Question Ids
                        filtered_im_question_ids_str = ",".join(
                            str(id) for id in filtered_im_question_ids
                        )

                        # mapping the cursor values to ImInternQuestionAnswer class
                        im_question_answer_list = process_rows(
                            read_cursor,
                            im_question_answers,
                            class_type=ImInternQuestionAnswer,
                        )

                        # fetching the qn_answers for baseline answers processing comparison
                        read_cursor.execute(
                            "select qa.* from qn_answers qa left join questions q on q.id = qa.QSN_ID WHERE q.IM_QUESTION_ID IN ({})".format(
                                filtered_im_question_ids_str
                            )
                        )
                        qn_answers = read_cursor.fetchall()

                        if check_list_is_valid(qn_answers):
                            questions_list = []

                            # mapping the cursor values to QuestionAnswer class
                            qn_answers_list = process_rows(
                                read_cursor, qn_answers, class_type=QuestionAnswer
                            )

                            # extract the ID column values from filtered_im_questions and store them as a new list using streams
                            qn_answer_ids = list(
                                map(lambda qn_answer: qn_answer.ID, qn_answers_list)
                            )

                            # Construct a comma-separated string of QuestionAnswer IDs
                            qn_answer_ids_str = ",".join(
                                str(id) for id in qn_answer_ids
                            )

                            read_cursor.execute(
                                "select q.* from questions q LEFT JOIN qn_answers qa on q.ID = qa.QSN_ID where qa.ID IN ({})".format(
                                    qn_answer_ids_str
                                )
                            )
                            questions = list(read_cursor.fetchall())

                            # mapping the cursor values to Question class
                            questions_list = process_rows(
                                read_cursor, questions, class_type=Question
                            )

                            # creating a map of Question Id and Imocha Question Id from questions
                            question_id_map = dict(
                                (question.ID, question.IM_QUESTION_ID)
                                for question in questions_list
                            )

                            score = 0
                            score_list = []
                            is_all_qns_processed = True

                            # iterating QuestionAnswer for corresponding baseline answer data for matching imocha question id
                            for im_question_answer in im_question_answer_list:
                                try:
                                    # building the s3 file key to fetch the audio file
                                    file_key = os.path.join(
                                        "atvis_files",
                                        str(im_test_invitation_id[0]),
                                        str(
                                            im_question_answer.INTERN_ASSMT_QUESTION_ID
                                        ),
                                        im_question_answer.S3_OBJECT_AUDIO_NAME,
                                    ).replace("\\", "/")
                                    file_path = download_s3_file(bucket_name, file_key)

                                    if file_path:
                                        for qn_answer in qn_answers_list:
                                            if im_assmt_question_map.get(
                                                im_question_answer.INTERN_ASSMT_QUESTION_ID
                                            ) == question_id_map.get(qn_answer.QSN_ID):
                                                user_recorded_answer = ""

                                                # Checking whether CUDA core supported graphics processor is available for Whisper processing
                                                devices = torch.device(
                                                    "cuda:0"
                                                    if torch.cuda.is_available()
                                                    else "cpu"
                                                )

                                                # Load the Whisper small model to transcribe the text from the recorded audio
                                                model = whisper.load_model(
                                                    "small", device=devices
                                                )

                                                # Transcribe the audio tensor using the Whisper model
                                                result = model.transcribe(file_path)

                                                user_recorded_answer = result["text"]

                                                print(
                                                    "Whisper generated answer is : ",
                                                    user_recorded_answer,
                                                )

                                                # load the spacy model that finds the similarity
                                                nlp = spacy_universal_sentence_encoder.load_model(
                                                    "en_use_lg"
                                                )

                                                actual_answer = qn_answer.PLAIN_TEXT
                                                print(
                                                    "Baseline answer is: ",
                                                    actual_answer,
                                                )

                                                user_output_document = nlp(
                                                    user_recorded_answer
                                                )
                                                actual_output_document = nlp(
                                                    actual_answer
                                                )

                                                accuracy = (
                                                    user_output_document.similarity(
                                                        actual_output_document
                                                    )
                                                )
                                                print(
                                                    "Accuracy (as-is) is : ", accuracy
                                                )
                                                print(
                                                    "Accuracy (as-is) in % is : ",
                                                    (accuracy * 100).__round__(2),
                                                )

                                                stop_words_removed_user_output_document = nlp(
                                                    remove_stopwords(
                                                        nlp, user_recorded_answer
                                                    )
                                                )
                                                stop_words_removed_actual_output_document = nlp(
                                                    remove_stopwords(nlp, actual_answer)
                                                )

                                                stop_words_removed_accuracy = stop_words_removed_user_output_document.similarity(
                                                    stop_words_removed_actual_output_document
                                                )
                                                score = (
                                                    stop_words_removed_accuracy * 100
                                                ).__round__(2)

                                                print(
                                                    "Accuracy (after stop words removed) is : ",
                                                    stop_words_removed_accuracy,
                                                )
                                                print(
                                                    "Accuracy (after stop words removed) in % is : ",
                                                    score,
                                                )

                                                # updating data into im_intern_questions_answer after processing with whisper open AI and spacy NLP
                                                write_cursor.execute(
                                                    update_questions_answer_query,
                                                    (
                                                        user_recorded_answer,
                                                        float(score),
                                                        im_question_answer.ID,
                                                    ),
                                                )
                                                write_connection.commit()

                                                score_list.append(score)

                                        # Delete the temporary file
                                        os.remove(file_path)

                                        if len(score_list) > 0:
                                            # will execute if there are multiple baseline answers present against a single question and finding the average accurate data out of it
                                            avg_score = statistics.mean(score_list)
                                            # updating data into ans_files after processing with whisper open AI and spacy NLP
                                            write_cursor.execute(
                                                update_questions_answer_query,
                                                (
                                                    user_recorded_answer,
                                                    float(avg_score),
                                                    im_question_answer.ID,
                                                ),
                                            )
                                            write_connection.commit()
                                            score_list = []

                                except s3_client.exceptions.NoSuchKey:
                                    print(
                                        f"File '{im_question_answer.S3_OBJECT_AUDIO_NAME}' not found in bucket '{bucket_name}'."
                                    )
                                    is_all_qns_processed = False
                                    continue
                                except Exception as e:
                                    print(f"Error during transcription: {str(e)}")
                                    is_all_qns_processed = False
                                    continue

                            if is_all_qns_processed:
                                # updating the status flag to indicate the server is complete state
                                write_cursor.execute(
                                    update_complete_status_query,
                                    (ai_processing_history.ID,),
                                )
                                write_connection.commit()

                            else:
                                return jsonify(
                                    {
                                        "message": "AI processing has been failed! Please try again"
                                    }
                                )

                    else:
                        print(
                            f"All Answers has been processed for the invitation {ai_processing_history.IM_HISTORY_ID}"
                        )

        else:
            return jsonify({"message": "No questions are available for ai processing"})

    else:
        return jsonify({"message": "All imocha histories are processed already"})

    # closing the database connections
    read_connection.close()
    write_connection.close()

    # return a JSON response indicating success
    return jsonify({"message": "Audio Files are Processed & Updated Successfully"})


# function to filter the stop words such as conjunction, preposition, articles etc.
def remove_stopwords(nlp, text):
    doc = nlp(text.lower())
    output = []
    for token in doc:
        if token.text in nlp.Defaults.stop_words:
            continue
        output.append(token.text)
    return " ".join(output)


def check_list_is_valid(input_list):
    return input_list is not None and len(input_list) > 0


# Define a generator function to map any list to defined class
def process_rows(read_cursor, cursor_results, class_type):
    custom_object_list = []
    for row in cursor_results:
        # dynamically create a dictionary of column names and values for this row
        column_names = [column[0] for column in read_cursor.description]
        row_dict = dict(zip(column_names, row))
        # create an instance of the given object using the dictionary
        class_object = class_type(**row_dict)
        custom_object_list.append(class_object)
    return custom_object_list


def download_s3_file(bucket_name, file_path):
    # Create a session using the provided access and secret keys
    session = boto3.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["AWS_SECRET_KEY"],
    )

    # Create an S3 client using the session
    s3_client = session.client("s3")

    # Get the S3 object URL
    s3_file_url = s3_client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket_name, "Key": file_path}
    )

    # Download the file from the URL to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name
        with requests.get(s3_file_url, stream=True) as download_request:
            download_request.raise_for_status()
            for chunk in download_request.iter_content(chunk_size=8192):
                temp_file.write(chunk)

    return temp_file_path


@app.route("/interview-bot/status/info", methods=["GET"])
def get_status():
    current_time = datetime.utcnow()

    response_data = {
        "data": {
            "appName": "InterviewBot Py",
            "gitBranch": "unknown",
            "gitHash": "unknown",
            "buildTime": current_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "buildNumber": "unknown",
            "version": "1.0.0",
        },
        "description": "Status info retrieved successfully",
        "statusCode": 200,
    }

    return jsonify(response_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
