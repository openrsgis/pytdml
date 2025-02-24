from io import BytesIO
import os
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv


class MinioConfig:
    def __init__(self, test_mode=False, server=None, access_key=None, secret_key=None):
        if test_mode:
            self._load_test_config()
        else:
            self._load_env_config(server, access_key, secret_key)

    def _load_test_config(self):
        self.server = "test_minio_server"
        self.access_key = "test_access_key"
        self.secret_key = "test_secret_key"

    def _load_env_config(self, server, access_key, secret_key):
        load_dotenv()
        self.server = server or os.getenv("MINIO_SERVER")
        self.access_key = access_key or os.getenv("MINIO_ACCESS_KEY")
        self.secret_key = secret_key or os.getenv("MINIO_SECRET_KEY")


class MinioClient:
    _instance = None

    def __new__(cls, config: MinioConfig):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.server = config.server
            cls._instance.access_key = config.access_key
            cls._instance.secret_key = config.secret_key
            cls._instance._create_client()
        return cls._instance

    def _create_client(self):
        try:
            self.client = Minio(self.server, access_key=self.access_key, secret_key=self.secret_key, secure=False)
        except S3Error as e:
            print("error:", e)

    def get_client(self):
        return self.client


test_config = MinioConfig(test_mode=True)
minio_client = MinioClient(test_config).client


class S3Client:
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
            # 列出存储桶中的对象
            response = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            # 打印对象列表

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


class LibraryNotInstalledError(Exception):
    pass
