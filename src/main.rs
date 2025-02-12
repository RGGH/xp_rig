use anyhow;
use futures::future::join_all;
use rig::completion::Prompt;
use rig::loaders::FileLoader;
use rig::providers::openai::Client;
use rig::providers::openai::GPT_35_TURBO_0125;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

//const IMAGE_URL: &str = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg";

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct MovieReview {
    sentiment: String,
    rating: f32,
    genre: String,
    actors: Vec<String>,
    era: u16,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ImageDetails {
    animal: bool,
    colour: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Tracing
    //tracing_subscriber::fmt()
    //.with_max_level(tracing::Level::DEBUG)
    //.with_target(false)
    //.init();

    let client = Client::from_env();

    // 1 ------------------------------------------------
    // structured entity recognition for movie review;
    //let extractor = client.extractor::<MovieReview>("gpt-4").build();
    //let review = extractor.extract("I loved this mid 90's road movie! It's a solid 9/10, Tom Cruise was good, and Samuel L Jackson was so cool.").await?;
    //println!("Extracted: {:?}", review);
    //

    // 2 ----- load file data, search image url, convert to structured output
    let res = file_load()?;
    println!("{:?}", res);

    // Create agent with a single context prompt to describe an image at a URL
    let agent = Arc::new(
        client
            .agent(GPT_35_TURBO_0125)
            .preamble("You are an image describer")
            .temperature(0.5)
            .build(),
    );

    // Compose `Image` for prompt
    //let image = IMAGE_URL;

    //// Prompt the agent and print the response with the structured extracted detail
    //let response = agent.prompt(image).await?;
    //println!("{}", response);
    //let extractor2 = client.extractor::<ImageDetails>("gpt-4").build();
    //let details = extractor2.extract(&response).await?;
    //println!("Extracted Image dtails {:?}", details);

    let tasks: Vec<_> = res
        .into_iter()
        .map(|image| {
            let agent = Arc::clone(&agent);
            let client_clone = client.clone();

            async move {
                let response = agent.prompt(image.as_str()).await.ok();
                let extractor2 = client_clone.extractor::<ImageDetails>("gpt-4").build();
                let details = extractor2.extract(response.as_deref().unwrap_or("")).await;
                if details.is_ok() {
                    Some(details)
                } else {
                    None
                }
            }
        })
        .collect::<Vec<_>>(); // Collect into Vec

    // Await all async tasks and print their results
    let results: Vec<_> = join_all(tasks).await;
    println!("{:?}", results);
    Ok(())
}

// load some text, (or URLS)  - then use to extract structured detail
fn file_load() -> Result<Vec<String>, anyhow::Error> {
    let res = FileLoader::with_glob("*.txt")?
        .read()
        .into_iter()
        .map(|r| r.map(|s| s.replace('\n', "")))
        .collect::<Result<Vec<_>, _>>()?; // Collect and propagate errors
    Ok(res)
}
