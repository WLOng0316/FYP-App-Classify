
package com.example.fyp_app1;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;

import com.example.fyp_app1.ml.Model;
import com.google.android.material.snackbar.Snackbar;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.provider.MediaStore;
import android.util.Log;
import android.view.View;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import android.widget.Button;
import android.widget.ImageView;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedHashMap;
import java.util.Map;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CAMERA_PERMISSION = 11;
    private static final int REQUEST_SELECT_IMAGE = 10;
    private static final int REQUEST_CAPTURE_IMAGE = 12;
    private static final String TAG = "MainActivity";

    Button selectBtn, predictBtn, captureBtn;
    TextView result;
    ImageView imageView;
    Bitmap bitmap;
    String[] labels = new String[1001];
    TableLayout nutritionTable;
    JSONObject localNutritionDatabase;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // permission
        getPermission();

        // Load local nutrition database
        localNutritionDatabase = loadLocalNutritionDatabase();

        int cnt = 0;
        try {
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(getAssets().open("labels.txt")));
            String line = bufferedReader.readLine();
            while (line != null) {
                labels[cnt] = line;
                cnt++;
                line = bufferedReader.readLine();
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        selectBtn = findViewById(R.id.selectBtn);
        predictBtn = findViewById(R.id.predictBtn);
        captureBtn = findViewById(R.id.captureBtn);
        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);
        nutritionTable = findViewById(R.id.nutrition_table);

        selectBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent();
                intent.setAction(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, REQUEST_SELECT_IMAGE);
            }
        });

        captureBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, REQUEST_CAPTURE_IMAGE);
            }
        });

        predictBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (bitmap == null) {
                    Log.e(TAG, "Bitmap is null, cannot proceed with prediction.");
                    result.setText("No image selected.");
                    return;
                }

                try {
                    Model model = Model.newInstance(MainActivity.this);

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);

                    bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

                    ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    Model.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    // Get the index of the maximum value in the output array
                    int maxIndex = getMax(outputFeature0.getFloatArray());

                    // Set the result text to the corresponding label
                    String predictedLabel = labels[maxIndex];
                    result.setText(predictedLabel);

                    // Fetch and display nutrition facts
                    fetchNutritionFacts(predictedLabel);

                    // Releases model resources if no longer used.
                    model.close();
                } catch (IOException e) {
                    Log.e(TAG, "Error during model inference", e);
                    result.setText("Error during prediction.");
                }
            }
        });
    }

    int getMax(float[] arr) {
        int maxIndex = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIndex]) maxIndex = i;
        }
        return maxIndex;
    }

    void getPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                getPermission();
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (requestCode == REQUEST_SELECT_IMAGE && data != null) {
                Uri uri = data.getData();
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                    imageView.setImageBitmap(bitmap);
                } catch (IOException e) {
                    Log.e(TAG, "Error loading image", e);
                    throw new RuntimeException(e);
                }
            } else if (requestCode == REQUEST_CAPTURE_IMAGE && data != null) {
                bitmap = (Bitmap) data.getExtras().get("data");
                imageView.setImageBitmap(bitmap);
            }
        }
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[224 * 224];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < 224; ++i) {
            for (int j = 0; j < 224; ++j) {
                int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
            }
        }
        return byteBuffer;
    }

    private JSONObject loadLocalNutritionDatabase() {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(getAssets().open("food_nutrition_database.json")));
            StringBuilder jsonString = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                jsonString.append(line);
            }
            return new JSONObject(jsonString.toString());
        } catch (IOException | JSONException e) {
            Log.e(TAG, "Error loading local nutrition database", e);
            return null;
        }
    }

    private void fetchNutritionFacts(String foodItem) {
        String endpointUrl = "https://trackapi.nutritionix.com/v2/natural/nutrients";
        String appId = "4d812ede";
        String apiKey = "aec831def9a08c4f66a8d99f4597322b";

        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.get("application/json; charset=utf-8");

        JSONObject json = new JSONObject();
        try {
            json.put("query", foodItem);
        } catch (JSONException e) {
            Log.e(TAG, "JSON error", e);
        }

        RequestBody body = RequestBody.create(json.toString(), JSON);
        Request request = new Request.Builder()
                .url(endpointUrl)
                .post(body)
                .addHeader("x-app-id", appId)
                .addHeader("x-app-key", apiKey)
                .addHeader("Content-Type", "application/json")
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                Log.e(TAG, "Nutrition API call failed", e);
                runOnUiThread(() -> displayNutritionInfo(getLocalNutritionInfo(foodItem)));
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (response.isSuccessful()) {
                    try {
                        String responseData = response.body().string();
                        JSONObject jsonResponse = new JSONObject(responseData);
                        JSONArray foods = jsonResponse.getJSONArray("foods");
                        if (foods.length() > 0) {
                            JSONObject foodData = foods.getJSONObject(0);
                            Map<String, String> necessaryInfo = new LinkedHashMap<>();
                            necessaryInfo.put("Food Name", foodData.optString("food_name", "N/A"));
                            necessaryInfo.put("Serving Quantity", String.valueOf(foodData.optInt("serving_qty", 0)));
                            necessaryInfo.put("Serving Unit", foodData.optString("serving_unit", "N/A"));
                            necessaryInfo.put("Serving Weight (grams)", String.valueOf(foodData.optInt("serving_weight_grams", 0)));
                            necessaryInfo.put("Calories", String.valueOf(foodData.optInt("nf_calories", 0)));
                            necessaryInfo.put("Total Fat (g)", String.valueOf(foodData.optInt("nf_total_fat", 0)));
                            necessaryInfo.put("Saturated Fat (g)", String.valueOf(foodData.optInt("nf_saturated_fat", 0)));
                            necessaryInfo.put("Cholesterol (mg)", String.valueOf(foodData.optInt("nf_cholesterol", 0)));
                            necessaryInfo.put("Sodium (mg)", String.valueOf(foodData.optInt("nf_sodium", 0)));
                            necessaryInfo.put("Carbohydrates (g)", String.valueOf(foodData.optInt("nf_total_carbohydrate", 0)));
                            necessaryInfo.put("Dietary Fiber (g)", String.valueOf(foodData.optInt("nf_dietary_fiber", 0)));
                            necessaryInfo.put("Sugars (g)", String.valueOf(foodData.optInt("nf_sugars", 0)));
                            necessaryInfo.put("Protein (g)", String.valueOf(foodData.optInt("nf_protein", 0)));
                            necessaryInfo.put("Potassium (mg)", String.valueOf(foodData.optInt("nf_potassium", 0)));
                            necessaryInfo.put("Phosphorus (mg)", String.valueOf(foodData.optInt("nf_p", 0)));

                            runOnUiThread(() -> displayNutritionInfo(necessaryInfo));
                        } else {
                            runOnUiThread(() -> displayNutritionInfo(getLocalNutritionInfo(foodItem)));
                        }
                    } catch (JSONException e) {
                        Log.e(TAG, "JSON parsing error", e);
                        runOnUiThread(() -> displayNutritionInfo(getLocalNutritionInfo(foodItem)));
                    }
                } else {
                    Log.e(TAG, "Unexpected response code: " + response.code());
                    runOnUiThread(() -> displayNutritionInfo(getLocalNutritionInfo(foodItem)));
                }
            }
        });
    }

    private Map<String, String> getLocalNutritionInfo(String foodItem) {
        Map<String, String> necessaryInfo = new LinkedHashMap<>();
        if (localNutritionDatabase != null) {
            try {
                if (localNutritionDatabase.has(foodItem)) {
                    JSONObject foodData = localNutritionDatabase.getJSONObject(foodItem);
                    necessaryInfo.put("Food Name", foodData.optString("food_name", "N/A"));
                    necessaryInfo.put("Serving Quantity", String.valueOf(foodData.optInt("serving_qty", 0)));
                    necessaryInfo.put("Serving Unit", foodData.optString("serving_unit", "N/A"));
                    necessaryInfo.put("Serving Weight (grams)", String.valueOf(foodData.optInt("serving_weight_grams", 0)));
                    necessaryInfo.put("Calories", String.valueOf(foodData.optInt("nf_calories", 0)));
                    necessaryInfo.put("Total Fat (g)", String.valueOf(foodData.optDouble("nf_total_fat", 0)));
                    necessaryInfo.put("Saturated Fat (g)", String.valueOf(foodData.optDouble("nf_saturated_fat", 0)));
                    necessaryInfo.put("Cholesterol (mg)", String.valueOf(foodData.optInt("nf_cholesterol", 0)));
                    necessaryInfo.put("Sodium (mg)", String.valueOf(foodData.optInt("nf_sodium", 0)));
                    necessaryInfo.put("Carbohydrates (g)", String.valueOf(foodData.optDouble("nf_total_carbohydrate", 0)));
                    necessaryInfo.put("Dietary Fiber (g)", String.valueOf(foodData.optDouble("nf_dietary_fiber", 0)));
                    necessaryInfo.put("Sugars (g)", String.valueOf(foodData.optDouble("nf_sugars", 0)));
                    necessaryInfo.put("Protein (g)", String.valueOf(foodData.optDouble("nf_protein", 0)));
                    necessaryInfo.put("Potassium (mg)", String.valueOf(foodData.optInt("nf_potassium", 0)));
                    necessaryInfo.put("Phosphorus (mg)", String.valueOf(foodData.optInt("nf_p", 0)));
                }
            } catch (JSONException e) {
                Log.e(TAG, "Error getting local nutrition info", e);
            }
        }
        return necessaryInfo;
    }

    private void displayNutritionInfo(Map<String, String> nutritionInfo) {
        nutritionTable.removeAllViews();

        TableRow headerRow = new TableRow(this);
        TextView headerNutrient = new TextView(this);
        TextView headerValue = new TextView(this);
        headerNutrient.setText("Nutrient");
        headerValue.setText("Value");
        headerNutrient.setTypeface(null, android.graphics.Typeface.BOLD);
        headerValue.setTypeface(null, android.graphics.Typeface.BOLD);
        headerNutrient.setPadding(8, 8, 8, 8);
        headerValue.setPadding(8, 8, 8, 8);
        headerNutrient.setBackgroundResource(R.drawable.border);
        headerValue.setBackgroundResource(R.drawable.border);
        headerRow.addView(headerNutrient);
        headerRow.addView(headerValue);
        headerRow.setBackgroundResource(R.drawable.border);
        nutritionTable.addView(headerRow);

        for (Map.Entry<String, String> entry : nutritionInfo.entrySet()) {
            TableRow row = new TableRow(this);
            TextView nutrient = new TextView(this);
            TextView value = new TextView(this);
            nutrient.setText(entry.getKey());
            value.setText(entry.getValue());
            nutrient.setPadding(8, 8, 8, 8);
            value.setPadding(8, 8, 8, 8);
            nutrient.setBackgroundResource(R.drawable.border);
            value.setBackgroundResource(R.drawable.border);
            row.addView(nutrient);
            row.addView(value);
            row.setBackgroundResource(R.drawable.border);
            nutritionTable.addView(row);
        }
    }

}