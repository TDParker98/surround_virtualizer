use hdf5;
use hound;
use hound::WavSpec;
use ndarray::s;
use std::f64::consts::PI;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sofa = "data/RIEC_hrir_subject_069.sofa";
    let (responses, response_rate) = sofa_r(sofa, vec![-90.0, 0.0, 1.5])?; // [0, 0, 1] is 1m in front of the listener
    println!("{:?}", response_rate);

    let r_file_name = "data/sample.wav";
    let (sample, spec) = wav_r(r_file_name)?;
    println!("{:?}", spec);

    let now = Instant::now();

    let output = virtualize(sample, responses)?;

    println!("\nRan in {} ms", now.elapsed().as_millis());

    let spec = WavSpec {
        channels: 2,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let w_file_name = "data/sample_out.wav";
    wav_w(w_file_name, spec, output)?;

    Ok(())
}

fn virtualize(
    sample: Vec<i16>,
    responses: Vec<Vec<f64>>,
) -> Result<Vec<Vec<i16>>, Box<dyn std::error::Error>> {
    let out_len = sample.len() + responses.iter().map(|x| x.len()).max().unwrap();
    let mut output: Vec<Vec<f64>> = vec![vec![0f64; out_len], vec![0f64; out_len]];
    for ii in 0..responses.len() {
        for jj in 0..sample.len() {
            let tt: Vec<f64> = responses[ii]
                .iter()
                .map(|x| x * sample[jj] as f64)
                .collect();
            for kk in 0..tt.len() {
                output[ii][jj + kk] = output[ii][jj + kk] + tt[kk];
            }
        }
    }

    let output: Vec<Vec<i16>> = output
        .into_iter()
        .map(|x| x.into_iter().map(|x| x as i16).collect())
        .collect();

    Ok(output)
}

fn sofa_r(
    file_name: &str,
    true_source: Vec<f64>,
) -> Result<(Vec<Vec<f64>>, u32), Box<dyn std::error::Error>> {
    let file = hdf5::File::open(file_name)?;

    // IDENTIFY ALL CATEGORIES //
    // https://www.sofaconventions.org/mediawiki/index.php/GeneralFIR //

    // let cats = file.datasets().unwrap();
    // for cat in cats {
    //     println!("{:?}", cat.name());
    // }

    let true_source_cart: Vec<f64> = vec![
        true_source[2]
            * (true_source[1] * (PI / 180.0)).cos()
            * -(true_source[0] * (PI / 180.0)).sin(), // x
        true_source[2]
            * (true_source[1] * (PI / 180.0)).cos()
            * (true_source[0] * (PI / 180.0)).cos(), // y
        true_source[2] * (true_source[1] * (PI / 180.0)).sin(), // z
    ];

    // println!("{:?}", file
    //     .dataset("ListenerView")?
    //     .read_2d::<f64>()?
    //     .slice(s![0, ..])
    //     .to_vec());

    let listener_pos = file
        .dataset("ListenerPosition")?
        .read_2d::<f64>()?
        .slice(s![0, ..])
        .to_vec();
    let sources = file.dataset("SourcePosition")?.read_dyn::<f64>()?;
    let mut sources_cart: Vec<Vec<f64>> = Vec::with_capacity(sources.dim()[0] as usize);
    for ii in 0..sources.dim()[0] {
        let source = sources.slice(s![ii, ..]).to_vec();
        sources_cart.push(vec![
            source[2] * (source[1] * (PI / 180.0)).cos() * -(source[0] * (PI / 180.0)).sin()
                - listener_pos[0],
            source[2] * (source[1] * (PI / 180.0)).cos() * (source[0] * (PI / 180.0)).cos()
                - listener_pos[1],
            source[2] * (source[1] * (PI / 180.0)).sin() - listener_pos[2],
        ])
    }

    let mut sources_norm: Vec<f64> = Vec::with_capacity(sources_cart.len() as usize);
    for source in sources_cart {
        sources_norm.push(
            ((true_source_cart[0] - source[0]).powf(2.0)
                + (true_source_cart[1] - source[1]).powf(2.0)
                + (true_source_cart[2] - source[2]).powf(2.0))
            .powf(0.5),
        )
    }

    let closest_index = sources_norm
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap();
    // println!("{:?}", &closest_index);
    // println!("{:?}", sources.slice(s![closest_index, ..]).to_vec());

    let response_rate = file.dataset("Data.SamplingRate")?.read_1d::<u32>()?[0];

    let delay = file
        .dataset("Data.Delay")?
        .read_2d::<usize>()?
        .slice(s![0, ..])
        .to_vec();

    let raw_responses = file.dataset("Data.IR")?.read_dyn::<f64>()?;

    let responses = vec![
        [
            vec![0f64; delay[0]],
            raw_responses.slice(s![closest_index, 0, ..]).to_vec(),
        ]
        .concat(),
        [
            vec![0f64; delay[1]],
            raw_responses.slice(s![closest_index, 1, ..]).to_vec(),
        ]
        .concat(),
    ];
    // _plot_vec(&responses[0], "plot")?;

    Ok((responses, response_rate))
}

// Reads a .wav file into a vector
fn wav_r(file_name: &str) -> Result<(Vec<i16>, WavSpec), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(file_name)?;
    let data: Vec<i16> = reader.samples::<i16>().map(|x| x.unwrap()).collect();
    let spec: WavSpec = reader.spec();

    Ok((data, spec))
}

// Writes a .wav file with a given vector
fn wav_w(
    file_name: &str,
    spec: WavSpec,
    data: Vec<Vec<i16>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = hound::WavWriter::create(file_name, spec)?;
    for ii in 0..data[0].len() {
        for jj in 0..spec.channels {
            writer.write_sample(data[jj as usize][ii])?;
        }
    }

    Ok(())
}

// Plots a given vector
fn _plot_vec(data_vec: &Vec<f64>, plot_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;

    let tt: Vec<i32> = (0..data_vec.len() as i32).collect();

    let mut data_tt: Vec<(i32, f64)> = Vec::with_capacity(data_vec.len() as usize);
    for ii in 0..data_vec.len() {
        data_tt.push((tt[ii], data_vec[ii]));
    }

    // PLOT //
    let plot_file = "data/".to_owned() + plot_name + ".png";
    let root = BitMapBackend::new(&plot_file, (5000, 1500)).into_drawing_area();

    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(plot_name, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0i32..data_vec.len() as i32, -0.75f64..0.75f64)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(data_tt, &RED))?;

    root.present()?;

    Ok(())
}
