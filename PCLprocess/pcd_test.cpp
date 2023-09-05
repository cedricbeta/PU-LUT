#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <boost/algorithm/string/split.hpp> // for split
#include <pcl/filters/statistical_outlier_removal.h>
#include <boost/filesystem.hpp>
#include <regex>
#include <string>

using namespace boost::filesystem;    
struct recursive_directory_range
{
    typedef recursive_directory_iterator iterator;
    recursive_directory_range(path p) : p_(p) {}

    iterator begin() { return recursive_directory_iterator(p_); }
    iterator end() { return recursive_directory_iterator(); }

    path p_;
};



using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

bool
loadCloud (const std::string &filename, PointCloud<PointXYZ>::Ptr cloud)
{
  std::ifstream fs;
  fs.open (filename.c_str (), std::ios::binary);
  if (!fs.is_open () || fs.fail ())
  {
    PCL_ERROR ("Could not open file '%s'! Error : %s\n", filename.c_str (), strerror (errno)); 
    fs.close ();
    return (false);
  }
  
  std::string line;
  std::vector<std::string> st;

  while (!fs.eof ())
  {
    std::getline (fs, line);
    // Ignore empty lines
    if (line.empty())
      continue;

    // Tokenize the line
    boost::trim (line);
    boost::split (st, line, boost::is_any_of ("\t\r "), boost::token_compress_on);

    if (st.size () != 3)
      continue;
    if (200.0f - atof (st[1].c_str ()) > 215.0f)
    {
      cloud->push_back (PointXYZ (static_cast<float>(atof (st[0].c_str ())), static_cast<float>(200.0f - atof (st[1].c_str ())), static_cast<float>(atof (st[2].c_str ()))));

    }
    
  }
  fs.close ();

  cloud->width = cloud->size (); cloud->height = 1; cloud->is_dense = true;
  return (true);
}


int
main ()
{
  

  uint16_t i = 0;
  for (auto it : recursive_directory_range("/home/chendong/PU-LUT/data/kinoptic_ptclouds/input/"))
  {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
      std::cout <<i<< " "<< it << std::endl;
      i++;
      
      if (!loadCloud (it.path().string(), cloud)) 
        return (-1);

      std::cout << "Loaded "
                << cloud->width * cloud->height
                << " data points from test_pcd.pcd with the following fields: "
                << std::endl;

      const std::string c_path = it.path().string();
      const std::string cf_path = std::regex_replace(c_path, std::regex("input"), "input_filtered");

      // Create the filtering object
      pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
      sor.setInputCloud (cloud);
      sor.setMeanK (50);
      sor.setStddevMulThresh (1.0);
      sor.filter (*cloud_filtered);

      pcl::io::savePCDFileASCII (cf_path, *cloud_filtered);

      // break;

      
  }


  // if (!loadCloud ("/home/v-chendwang/test-playground/dataset/kinoptic_ptclouds/training/input_2X/input_ld/ptcloud_hd00000500.xyz", cloud)) 
  //   return (-1);

  // std::cout << "Loaded "
  //           << cloud->width * cloud->height
  //           << " data points from test_pcd.pcd with the following fields: "
  //           << std::endl;

  // for (const auto& point: *cloud)
  //   std::cout << "    " << point.x
  //             << " "    << point.y
  //             << " "    << point.z << std::endl;


  // pcl::io::savePCDFileASCII ("test_pcd.pcd", *cloud);

  // // Create the filtering object
  // pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  // sor.setInputCloud (cloud);
  // sor.setMeanK (50);
  // sor.setStddevMulThresh (1.0);
  // sor.filter (*cloud_filtered);

  // std::cerr << "Cloud after filtering: " << std::endl;
  // std::cerr << *cloud_filtered << std::endl;

  // pcl::io::savePCDFileASCII ("test_pcd_filtered.pcd", *cloud_filtered);

  return (0);
}