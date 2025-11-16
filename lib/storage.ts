import { Storage } from '@google-cloud/storage';

let storage: Storage | null = null;

export function getStorageClient() {
  if (!storage) {
    storage = new Storage({
      projectId: process.env.GOOGLE_CLOUD_PROJECT_ID,
      keyFilename: process.env.GOOGLE_APPLICATION_CREDENTIALS,
    });
  }
  return storage;
}

export async function uploadToStorage(
  file: Buffer,
  fileName: string,
  bucketName: string,
  contentType: string = 'audio/mpeg'
): Promise<string> {
  const storage = getStorageClient();
  const bucket = storage.bucket(bucketName);
  const blob = bucket.file(fileName);

  await blob.save(file, {
    contentType,
    metadata: {
      cacheControl: 'no-cache',
    },
  });

  return `gs://${bucketName}/${fileName}`;
}

export async function getSignedUrl(
  bucketName: string,
  fileName: string,
  expiresInMinutes: number = 60
): Promise<string> {
  const storage = getStorageClient();
  const bucket = storage.bucket(bucketName);
  const file = bucket.file(fileName);

  const [url] = await file.getSignedUrl({
    version: 'v4',
    action: 'read',
    expires: Date.now() + expiresInMinutes * 60 * 1000,
  });

  return url;
}

export async function deleteFile(
  bucketName: string,
  fileName: string
): Promise<void> {
  const storage = getStorageClient();
  const bucket = storage.bucket(bucketName);
  await bucket.file(fileName).delete();
}

export async function fileExists(
  bucketName: string,
  fileName: string
): Promise<boolean> {
  const storage = getStorageClient();
  const bucket = storage.bucket(bucketName);
  const [exists] = await bucket.file(fileName).exists();
  return exists;
}
