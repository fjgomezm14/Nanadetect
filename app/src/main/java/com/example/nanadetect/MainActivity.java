package com.example.nanadetect;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import com.example.nanadetect.ml.Newmodel1;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_IMAGE_CAPTURE = 101;
    private static final int REQUEST_IMAGE_PICK = 102;
    private ImageView imageView;
    private int imageSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView4);
        Button buttonPicture = findViewById(R.id.buttonpicture);
        Button buttonUpload = findViewById(R.id.buttonupload);
        ImageView aidIcon = findViewById(R.id.aidicon);
        Button treatmentButton = findViewById(R.id.treatmentbutton);

        buttonPicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                dispatchTakePictureIntent();
            }
        });

        buttonUpload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                openImagePicker();
            }
        });

        // Add onClickListener for the aidicon ImageView
        aidIcon.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Handle the click event, e.g., navigate to Screen2
                Intent intent = new Intent(MainActivity.this, Screen2.class);
                startActivity(intent);
            }
        });

        // Add onClickListener for the Treatment button
        treatmentButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Get the class name from the existing TextView
                TextView imageName = findViewById(R.id.imageName);
                String className = imageName.getText().toString();


                // Open corresponding activity based on the identified leaf disease
                switch (className) {
                    case "Black Sigatoka": {
                        // Open screen3.java
                        Intent intent = new Intent(MainActivity.this, Screen3.class);
                        startActivity(intent);
                        break;
                    }
                    case "Fusarium Wilt": {
                        // Open screen4.java
                        Intent intent = new Intent(MainActivity.this, Screen4.class);
                        startActivity(intent);
                        break;
                    }
                    case "Healthy Leaf": {
                        // Open screen5.java
                        Intent intent = new Intent(MainActivity.this, Screen5.class);
                        startActivity(intent);
                        break;
                    }
                    case "Moko Disease": {
                        // Open screen6.java
                        Intent intent = new Intent(MainActivity.this, Screen6.class);
                        startActivity(intent);
                        break;
                    }
                    case "Banana Bunchy Top": {
                        // Open screen7.java
                        Intent intent = new Intent(MainActivity.this, Screen7.class);
                        startActivity(intent);
                        break;
                    }
                }

                // Add more conditions if needed for other diseases
            }
        });
    }

    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        }
    }

    private void openImagePicker() {
        Intent pickImageIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(pickImageIntent, REQUEST_IMAGE_PICK);
    }

    public void classifyImage(Bitmap image) {
        try {
            Newmodel1 model = Newmodel1.newInstance(getApplicationContext());

            // Create inputs for the model.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, imageSize, imageSize, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            int pixel = 0;
            for (int i = 0; i < imageSize; ++i) {
                for (int j = 0; j < imageSize; ++j) {
                    int val = intValues[pixel++];
                    float red = ((val >> 16) & 0xFF) * (1.f / 255.f);
                    float green = ((val >> 8) & 0xFF) * (1.f / 255.f);
                    float blue = (val & 0xFF) * (1.f / 255.f);
                    byteBuffer.putFloat(red);
                    byteBuffer.putFloat(green);
                    byteBuffer.putFloat(blue);
                }
            }

            // Load the image data into the input buffer.
            inputFeature0.loadBuffer(byteBuffer);

            // Run model inference and get results.
            Newmodel1.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] class_names = {"Banana Bunchy Top", "Black Sigatoka", "Fusarium Wilt", "Healthy Leaf", "Moko Disease"};
            String className = class_names[maxPos];

            // Display the class name in the TextView.
            TextView imageName = findViewById(R.id.imageName);
            imageName.setText(className);

            // Display the class name in the TextView.
            TextView imageName2 = findViewById(R.id.imagename2);

            // Calculate the confidence rate based on maxPos.
            float confidenceRate = maxConfidence * 100;

            if (confidenceRate == 0.0) {
                // If confidenceRate is 0.0%, set the text of the existing imageName TextView.
                TextView imageNameTextView = findViewById(R.id.imageName);
                imageNameTextView.setText(R.string.try_another_image);
                imageName2.setText(R.string.not_banana_leaf);
            } else {
                // Set the text of the existing imageName TextView with the class name.
                imageName.setText(className);
                imageName2.setText(R.string.banana_leaf);
            }

            // Display the confidence rate in the TextView.
            TextView confidenceRateTextView = findViewById(R.id.confidenceRate);
            confidenceRateTextView.setText(String.format("%.1f%%", confidenceRate));

            // Release model resources.
            model.close();
        } catch (IOException e) {
            e.printStackTrace();
            // Handle the exception, e.g., by showing an error message to the user.
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK) {
            if (requestCode == REQUEST_IMAGE_CAPTURE && data != null) {
                Bundle extras = data.getExtras();
                if (extras != null) {
                    Bitmap imageBitmap = (Bitmap) extras.get("data");
                    processImage(imageBitmap);
                }
            } else if (requestCode == REQUEST_IMAGE_PICK && data != null) {
                Uri imageUri = data.getData();
                try {
                    Bitmap selectedImage = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                    processImage(selectedImage);
                } catch (IOException e) {
                    e.printStackTrace();
                    // Handle the exception, e.g., by showing an error message to the user.
                }
            }
        }
    }

    private void processImage(Bitmap image) {
        imageView.setImageBitmap(image);
        Bitmap resizedImage = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
        classifyImage(resizedImage);
    }

}