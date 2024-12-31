from io import BytesIO
import re


def parse_s3_path(s3_path):
    # Match and extract the bucket and key using regular expressions
    pattern = r'^s3://([^/]+)/(.+)$'
    match = re.match(pattern, s3_path)
    if match:
        bucket_name = match.group(1)
        key = match.group(2)
        return bucket_name, key
    else:
        raise ValueError('Invalid S3 path')


class LibraryNotInstalledError(Exception):
    pass


class S3Client:
    """
    Read TrainingDML-AI encoded data from S3 object storage.
    """

    def __init__(self, resource, server, access_key, secret_key):
        try:
            import boto3
        except ModuleNotFoundError:
            raise LibraryNotInstalledError("Failed to import boto3, please install the library first")
        try:
            self.s3_client = boto3.client(resource, endpoint_url=server, aws_access_key_id=access_key,
                                          aws_secret_access_key=secret_key)
        except Exception as e:
            print("Exception when connecting to S3: ", str(e))

    def list_buckets(self):
        response = self.s3_client.list_buckets()
        # Print storage bucket name
        bucket_list = []
        for bucket in response['Buckets']:
            bucket_list.append(bucket)
        return bucket_list

    def list_objects(self, bucket_name, prefix):
        obj_list = []
        try:
            # List the objects in the bucket
            response = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            # Print object list

            if 'Contents' in response:
                for obj in response['Contents']:
                    obj_list.append(obj['Key'])
            else:
                print("There are no objects in the storage bucket: " + bucket_name)

        except Exception as e:
            print("An exception occurs when listing objects: ", str(e))
        finally:
            return obj_list

    def get_object(self, bucket, key):
        response = self.s3_client.get_object(Bucket=bucket, Key=key)

        # Get object content
        object_data = response['Body'].read()
        return BytesIO(object_data)

    def download_file(self, bucket_name, object_key, file_path):
        try:
            # Download objects to local files
            self.s3_client.download_file(bucket_name, object_key, file_path)

            print("The object has been successfully downloaded to the file:", file_path)

        except Exception as e:
            print("An exception occurred while downloading an object: ", str(e))


