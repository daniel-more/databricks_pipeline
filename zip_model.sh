mkdir upload_bundle
cp -r models upload_bundle/
cp -r stages upload_bundle/
cp input_example.parquet upload_bundle/models/gbt_model/
cp model_signature.pkl upload_bundle/models/gbt_model/

zip -r upload_bundle.zip upload_bundle
