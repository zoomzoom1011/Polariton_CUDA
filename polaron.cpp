//CUDA version of polaron code from Dr. Zoom
// this code require CUDA-aware MPI technology from Nvidia

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cstring>
#include <time.h>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <iomanip>
#include <unistd.h>
#include <algorithm>    // std::min
#include <sys/stat.h>
#include <dirent.h>
#include <chrono> 
#include <mpi.h>  

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"

using namespace std;
using namespace std::chrono; 

#define MASTER 0

// defining the global variable
string task_title;
int nmax = 2;
int vibmax, sys_vibmax;
double hw;
string xyz_file[3];
int lattice_x = 1;
int lattice_y = 1;
int lattice_z = 1;
double jo_x; 
double jo_y; 
double jo_z; 
double s  = 1.0;
double s_cat = 0.0;
double s_ani = 0.0;
double abs_lw;
double spec_start_pl, spec_end_pl;
bool lorentzian = false;
bool no_frenkel = false;
bool calc_pl, nearest_neighbor;


double *sys_h = NULL; //in CPU


// counters
int kount = 0;
int kount_1p = 0;
int kount_2p = 0;
int kount_lattice = 0;

// class method
class basis_1p
{
public:
    int lx;
    int ly;
    int lz;
    int vib;
};

class basis_2p
{
public:
    int lx;
    int ly;
    int lz;
    int vib;
    int lxv;
    int lyv;
    int lzv;
    int vibv;
};

// indexes
int* nx_lattice = NULL; 
basis_1p* nx_1p = NULL; 
basis_2p* nx_2p = NULL;
double* disorder = NULL;   

// franck condon factors
// double* fc_gf = NULL;

// tdm
double* ux = NULL;
double* uy = NULL;
// double* uz = NULL;

// constant 
const double pi = atan(1.00) * 4;
const double ev = 8065.0; 
const double hc = 1.23984193*pow(10,3)*ev;  //plancks constant times the speed of light in nm*hw
const double kb = 0.6956925;         //units of cm-1 k

// multiparticle states
bool one_state = true; 
bool two_state = false; 
bool LL = true; 
bool SS = false;
bool LS = false;

// periodic
bool periodic = true;
    
// absorption spectra
int spec_step;
double spec_start_ab, spec_end_ab;
// double* sys_eval = NULL; 
bool abs_freq_dep; 
double* ab_osc_x = NULL;
double* ab_osc_y = NULL;
double* ab_osc_z = NULL;      
double* ab_x = NULL;   
double* ab_y = NULL; 
double* mab_x = NULL;   
double* mab_y = NULL;   
double* ab_z = NULL;    

// pl spectra
double* pl_osc_x = NULL;
double* pl_osc_y = NULL;
double* pl_osc_z = NULL;
double* pl_x = NULL;  
double* pl_y = NULL;  
double* pl_z = NULL;   
int pl_start; 

// geometry
double dielectric = 1.0;
bool extended_cpl = false; 
double sigma;
int conf_max;


//use input name to get output name
string FileName(string name)
{
    int site = name.find_last_of('.');
    if (site == 0)
    {
        name = "output";
    }
    else if (site > 0)
    {
        name.erase(site, name.size() - site + 1);
    }
    return name;
}


//host name function
string HostName()
{
    string hostname;
    //for linux system
    ifstream hostname_file("/etc/hostname", ifstream::in);
    hostname_file >> hostname;
    return hostname;
}

double get_j(int lx1, int lx2, int ly1, int ly2, int lz1, int lz2) {
    int dx, dy, dz;
    double temp = 0.0; 

    dx = abs( lx1 - lx2 );
    dy = abs( ly1 - ly2 );
    dz = abs( lz1 - lz2 );
    if (periodic){
        dx = min( dx, lattice_x - dx );
        dy = min( dy, lattice_y - dy );
        dz = min( dz, lattice_z - dz );
    }

    if ( !extended_cpl ){
        if ( dy == 0 && dz == 0 ){
            if (dx == 1 ) temp = jo_x;
        }
        if ( dx == 0 && dz == 0 ){
            if ( dy == 1 ) temp = jo_y;
        }
        if ( dx == 0 && dy == 0 ){
            if ( dz == 1 ) temp = jo_z;
        }
    }
    // else if ( extended_cpl ){
        // temp = jo_ex( dx, dy, dz )
    // }

    return temp;
}

double fact(int n){
    double temp = 1.0;
    int i;
    
    if (n >= 0){
        for( i = 2;i <= n; i++){
            temp *= i; 
        }
    }
    return temp; 
}

double fcfac(int n, int m, double s){
    double fc = 0.0; 
    double f_m,f_n,f_k,f_nmk,f_mk,facin; 
    int k; 
    
    for(k = 0; k <= m; k++){
        if(n-m+k<0) continue; 
        
		f_mk  = fact(m-k);
		f_nmk = fact(n-m+k);
		f_k   = fact(k);
		facin = 1.0/(f_k*f_mk*f_nmk);

		fc += facin*pow(s,(k*0.50))*pow(s,((n-m+k)*0.50))*pow(-1,(n-m+k)); 
    }
	f_n = fact(n);
	f_m = fact(m);
	fc *= sqrt(f_m*f_n) * exp(-s/2.0);
    
    return fc; 
}
double fc_gf(int vib1,  int vib2){
    double temp;
    temp = fcfac(vib1, vib2, s); 
    
    return temp; 
}

// double get_distance(int n1, int da1, int n2, int da2) {
    // double distance = 0.0;
    // int i = 9*n1 + 3*da1 - 12;
    // int j = 9*n2 + 3*da2 - 12;
    // distance = pow((mol1pos[i]-mol1pos[j]),2)
                  // +pow((mol1pos[i+1]-mol1pos[j+1]),2)
                  // +pow((mol1pos[i+2]-mol1pos[j+2]),2); 
    // distance = sqrt(distance);

    // return distance;
// }



double factorial(int n) {
    double factorial_1 = 1.0;
    if (n < 0) {
        cout << "Factorial not calculatable for: " << n << endl;
        exit(0);
    }
    else {
        if (n != 0) {
            for (int i = 2; i <= n; i++) {
                factorial_1 = factorial_1 * i;
            }
        }
    }
    return factorial_1;
}
__global__ void abs_osc  (int sys_kount, double d_sys_h[], 
    double d_ux[], double d_uy[], double d_ab_osc_x[], double d_ab_osc_y[]); 
__global__ void abs_spectra (int sys_kount, double d_ab_osc_x[], double d_ab_osc_y[], double d_sys_eval[], 
    double d_ab_x[], double d_ab_y[], 
    int spec_step, double spec_start_ab, double spec_end_ab, double abs_lw ); 
void magma_diagonalization( int sys_kount, double sys_h[], 
    double ux[], double uy[], double ab_x[], double ab_y[], 
    int spec_step, double spec_start_ab, double spec_end_ab, double abs_lw); 

int main(int argc, char** argv) {
    int state, n1, n2, sysnx, da1, da2, state1, state2, hx, spec_point, run, n;
    double energy, tran_e, lineshape;
    int vib, lx, ly, lz;
    int vibv, lxv, lyv, lzv; 
    int vib1, lx1, ly1, lz1; 
    int vib2, lx2, ly2, lz2; 
    int vib1v, lx1v, ly1v, lz1v; 
    int vib2v, lx2v, ly2v, lz2v; 
    int i, j, k; 
    int lxyz; 
    int config; 
    // const int blockSize = 128;    // The number of threads to launch per block, better to be the times of 64

    int kount2; 

    int myid;   // processor index 
    int numprocs; // #processor

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Barrier(MPI_COMM_WORLD);

    int iwork1 = conf_max / numprocs;
    int iwork2 = conf_max % numprocs;
    int ista = myid * iwork1 + 1 + min(myid, iwork2);
    int iend = ista + iwork1 - 1;
    if ( iwork2 > myid) iend = iend + 1;

    time_t start=time(NULL), end;

    magma_print_environment();

    // GPU variables                 

    // ***              Variable Declaration            *** //
    // **************************************************** //

    //input file check
    if ( argc != 2 ){
        printf("Usage:\n"
               "\tInclude as the first argument either the name of an input file,  or a checkpoint\n"
               "\tfile with extension '.cpt' if restarting the calculation. No other arguments are\n"
               "\tallowed.\n");
        exit(EXIT_FAILURE);   
    }

    // retrieve and print info about gpu
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop,0);
    // printf("\nGPU INFO:\n"
    //        "\tDevice name: %s\n"
    //        "\tMemory: %g gb\n",
    //        prop.name, prop.totalGlobalMem/(1.E9));
    
    // // register signal handler to take care of interruption and termination signals
    // signal( SIGINT,  signal_handler );
    // signal( SIGTERM, signal_handler );

    printf("\n>>> Setting parameters\n");
    
    std::ifstream file(argv[1],ifstream::in);
    if (!file)
    {
        cerr << "ERROR: unable to open input file: " << argv[1] << endl;
        exit(2);
    }
    string buff;
    string label;

    if (file.is_open()) {
        string line;
        
        while (getline(file, line)) {
            // using printf() in all tests for consistency
            if (line[0] == '#') continue;

            for (i = 0; i <= line.length(); i++)
            {
                if (line[i] == ' ') {
                    break;
                }    
            }

            for (j = i; j <= line.length(); j++)
            {
                if (line[j] != ' ') {
                    break;
                }
            }

            for (k = j; k <= line.length(); k++)
            {
                if (line[k] == ' ') {
                    break;
                }
            }
            label = line.substr(0, i);
            buff = line.substr(j, k-j);

            if (label == "task_title") {
                task_title = buff;
                cout << "setting task_title to:" << task_title << endl;
            }
            else if (label == "lattice_x") {
                sscanf(buff.c_str(), "%d", &lattice_x);
                cout << "Setting lattice_x to: " << lattice_x << endl;
            }
            else if (label == "lattice_y") {
                sscanf(buff.c_str(), "%d", &lattice_y);
                cout << "Setting lattice_y to: " << lattice_y << endl;
            }
            else if (label == "lattice_z") {
                sscanf(buff.c_str(), "%d", &lattice_z);
                cout << "Setting lattice_z to: " << lattice_z << endl;
            }
            else if (label == "vibmax") {
                sscanf(buff.c_str(), "%d", &vibmax);
                cout << "Setting vibmax to: " << vibmax << endl; 
            }
            else if (label == "hw") {
                sscanf(buff.c_str(), "%lf", &hw);
                cout << "Setting vibration energy to: " << hw << endl;  
            }
            else if (label == "jo_x") {
                sscanf(buff.c_str(), "%lf", &jo_x);
                cout << "Setting jo_x to: " << jo_x << endl;
            }
            else if (label == "jo_y") {
                sscanf(buff.c_str(), "%lf", &jo_y);
                cout << "Setting jo_y to: " << jo_y << endl;
            }
            else if (label == "jo_z") {
                sscanf(buff.c_str(), "%lf", &jo_z);
                cout << "Setting jo_z to: " << jo_z << endl;
            }
            else if (label == "calc_pl") {
                if (buff == ".true."){
                    calc_pl = true;
                    cout << "Will calculate all spectra " << endl;
                } else if(buff == ".false."){
                    calc_pl = false;
                    cout << "Will calculate only absorption " << endl;
                }
            }
            else if (label == "lorentzian") {
                if (buff == ".true."){
                    lorentzian = true;
                    cout << "Lineshape set to Lorentzian " << endl;
                } else if(buff == ".false."){
                    lorentzian = false;
                    cout << "Lineshape set to Gaussian " << endl;
                }
            }
            else if (label == "periodic") {
                if (buff == ".true.") {
                    periodic = true;
                    cout << "periodic condition is on " << endl;
                }
                else if (buff == ".false.") {
                    periodic = false;
                    cout << "periodic condition is off " << endl;
                }
                
            }
            else if (label == "nearest_neighbor") {
                if (buff == ".true.") {
                    nearest_neighbor = true;
                    cout << "calc coupling from nearest neighbor " << endl;
                }
                else if (buff == ".false.") {
                    nearest_neighbor = false;
                    cout << "calc coupling from long range " << endl;
                }
                
            }
            else if (label == "two_state") {
                if (buff == ".true.") {
                    two_state = true;
                    cout << "will use two state" << endl;
                }
                else if (buff == ".false.") {
                    two_state = false;
                    cout << "won't use two state" << endl;
                }
                
            }
            else if (label == "s") {
                sscanf(buff.c_str(), "%lf", &s);
                cout << "Setting s to: " << s << endl;
                
            }
            else if (label == "dielectric") {
                sscanf(buff.c_str(), "%lf", &dielectric);
                cout << "Setting dielectric to: " << dielectric << endl;
                
            }
            else if (label == "abs_lw") {
                sscanf(buff.c_str(), "%lf", &abs_lw);
                cout << "Setting the abs linewidth to (cm-1): " << abs_lw << endl;
                
            }
            else if (label == "spec_step") {
                sscanf(buff.c_str(), "%d", &spec_step);
                cout << "Setting spec_step to (cm-1): " << spec_step << endl;
                
            }
            else if (label == "spec_start_ab") {
                sscanf(buff.c_str(), "%lf", &spec_start_ab);
                cout << "Setting spec_start_ab to: " << spec_start_ab << endl;
                
            }
            else if (label == "spec_end_ab") {
                sscanf(buff.c_str(), "%lf", &spec_end_ab);
                cout << "Setting spec_end_ab to: " << spec_end_ab << endl;
                
            }
            else if (label == "spec_start_pl") {
                sscanf(buff.c_str(), "%lf", &spec_start_pl);
                cout << "Setting spec_start_pl to: " << spec_start_pl << endl;
                
            }
            else if (label == "spec_end_pl") {
                sscanf(buff.c_str(), "%lf", &spec_end_pl);
                cout << "Setting spec_end_pl to: " << spec_end_pl << endl;
                
            }
            else if (label == "conf_max") {
                sscanf(buff.c_str(), "%d", &conf_max);
                cout << "Setting conf_max to : " << conf_max << endl;
                
            }
            else if (label == "sigma") {
                sscanf(buff.c_str(), "%lf", &sigma);
                cout << "Setting sigma to: " << sigma << endl;
                
            }
            // else if (label == "xyz_file1") {
            //     xyz_file[0] = buff;
            //     cout << "Will read the xyz file: " << xyz_file[0] << endl;
                
            // }
            // else if (label == "xyz_file2") {
            //     xyz_file[1] = buff;
            //     cout << "Will read the xyz file: " << xyz_file[1] << endl;
                
            // }
            //  else if (label == "xyz_file3") {
            //      xyz_file[2] = buff;
            //      cout << "Will read the xyz file: " << xyz_file[2] << endl;
                
            // }
            else
                cout << "invalid label at line, " << label << buff << endl;
                //abort();
        }
        file.close();
    }
// //**********************************************//
    //change to working directory
// //**********************************************//
    int status = mkdir(task_title.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    chdir(task_title.c_str());

// //**********************************************//
    // index lattice
// //**********************************************//
    kount_lattice = lattice_x*lattice_y*lattice_z; 
    
    nx_lattice = new int[kount_lattice];
    for (lxyz = 1; lxyz <= kount_lattice; lxyz++) {    //x,y,z
        nx_lattice[lxyz-1] = lxyz; 
    }
    
    for (lx = 1; lx <= lattice_x; lx++) {    //x
    for (ly = 1; ly <= lattice_y; ly++) {    //y
    for (lz = 1; lz <= lattice_z; lz++) {    //z
        i = (lx-1) * lattice_y * lattice_z + (ly-1) * lattice_z + (lz-1); 
        nx_lattice[i] = i+1; 
    }
    }
    }
    
// //**********************************************//
// //                        index the 1 p states                                   //
// //**********************************************//
    for (run = 1; run <= 2; run++) {
        kount_1p = 0; 
        for (lx = 1; lx <= lattice_x; lx++) {    //x
        for (ly = 1; ly <= lattice_y; ly++) {    //y
        for (lz = 1; lz <= lattice_z; lz++) {    //z
            i = (lx-1) * lattice_y * lattice_z + (ly-1) * lattice_z + (lz-1); 
            lxyz = nx_lattice[i]; 
            for (vib = 0; vib <= vibmax; vib++) {
                kount_1p += 1;
                
                if (run == 2){
                    nx_1p[kount_1p - 1].lx = lx;
                    nx_1p[kount_1p - 1].ly = ly;
                    nx_1p[kount_1p - 1].lz = lz;
                    nx_1p[kount_1p - 1].vib = vib;
                }
            }
        }
        }
        }
        if (run == 1) {
            nx_1p = new basis_1p[kount_1p];
        }
    }
// //**********************************************//
// //                        index the 2 p states                                   //
// //**********************************************//
    if ( two_state ) {
        for (run = 1; run <= 2; run++) {
        kount_2p = 0; 
            for (lx = 1; lx <= lattice_x; lx++) {    //x
            for (ly = 1; ly <= lattice_y; ly++) {    //y
            for (lz = 1; lz <= lattice_z; lz++) {    //z
            for (vib = 0; vib <= vibmax; vib++) {
            for (lxv = 1; lxv <= lattice_x; lxv++) {    //x
            for (lyv = 1; lyv <= lattice_y; lyv++) {    //y
            for (lzv = 1; lzv <= lattice_z; lzv++) {    //z
            for (vibv = 1; vibv <= vibmax; vibv++) {
                if ( lx == lxv && ly == lyv && lz == lzv ) continue; 
                if ( vib+vibv > vibmax ) continue; 
                
                kount_2p += 1;
                
                if (run == 2){
                    nx_2p[kount_2p - 1].lx = lx;
                    nx_2p[kount_2p - 1].ly = ly;
                    nx_2p[kount_2p - 1].lz = lz;
                    nx_2p[kount_2p - 1].vib = vib;
                    nx_2p[kount_2p - 1].lxv = lxv;
                    nx_2p[kount_2p - 1].lyv = lyv;
                    nx_2p[kount_2p - 1].lzv = lzv;
                    nx_2p[kount_2p - 1].vibv = vibv;
                    // cout << lx << vib << lxv << vibv << endl; 
                }
            }
            }
            }
            }
            }
            }
            }
            }
        if (run == 1) {
            nx_2p = new basis_2p[kount_2p];
        }
    }
//***********************************************/    
    // output the kount
//***********************************************/
    kount = kount_1p + kount_2p;

    cout << "********************************* " << endl;
    cout << "kount   : " << kount << endl;
    cout << "kount 1p: " << kount_1p << endl;
    cout << "kount 2p: " << kount_2p << endl;
    cout << "********************************* " << endl;

//***********************************************/
    // create the Hamiltonian  
//***********************************************/  
    kount2 = kount*kount; 
    sys_h = new double[kount2]; 

/***********************************************/
    // transition dipole moment
//***********************************************/
    ux = new double[kount];
    uy = new double[kount];
    // uz = new double[sys_kount];


    // go over all 1p basis states
    for (i = 0; i < kount_1p; i++) {  
        ux[i] = 0.0; 
        uy[i] = 0.0;
        // uz[i] = 0.0; 

        lx1 = nx_1p[i].lx;
        ly1 = nx_1p[i].ly;

        ux[i] = lx1; 
        uy[i] = ly1;
    }

	// go over all 2p basis states
    for (i = 0; i < kount_2p; i++) {  

        ux[i+kount_1p] = 0.0; 
        uy[i+kount_1p] = 0.0;

        lx1 = nx_2p[i].lx;
        ly1 = nx_2p[i].ly;

        ux[i+kount_1p] = lx1; 
        uy[i+kount_1p] = ly1;
    }

/***********************************************/
//                        create 1p1p hamiltonian                                 //
//***********************************************/
    for (i = 0; i < kount_1p; i++) {  
        lx1 = nx_1p[i].lx;
        ly1 = nx_1p[i].ly;
        lz1 = nx_1p[i].lz;
        vib1 = nx_1p[i].vib;
        
        sys_h[i * kount + i] = vib1 * hw; 
        
        for (j = 0; j < kount_1p; j++) {  
            if(i == j) continue; 
            
            lx2 = nx_1p[j].lx;
            ly2 = nx_1p[j].ly;
            lz2 = nx_1p[j].lz;
            vib2 = nx_1p[j].vib; 
            
            sys_h[i * kount + j] = get_j(lx1, lx2, ly1, ly2, lz1, lz2)*
                    fc_gf(0, vib1) * fc_gf(0, vib2); 
            sys_h[j * kount + i] = sys_h[i * kount + j]; 
        }
    }  
    
    
//***********************************************//
//                        create 1p2p hamiltonian                                 //
//***********************************************/
    for (i = 0; i < kount_1p; i++) {  
        lx1 = nx_1p[i].lx;
        ly1 = nx_1p[i].ly;
        lz1 = nx_1p[i].lz;
        vib1 = nx_1p[i].vib;
        
        for (j = 0; j < kount_2p; j++) {  
            
            lx2 = nx_2p[j].lx;
            ly2 = nx_2p[j].ly;
            lz2 = nx_2p[j].lz;
            vib2 = nx_2p[j].vib; 
            lx2v = nx_2p[j].lxv;
            ly2v = nx_2p[j].lyv;
            lz2v = nx_2p[j].lzv;
            vib2v = nx_2p[j].vibv; 
            
            if(lx2v == lx1 && ly2v == ly1 && lz2v == lz1){
                sys_h[i * kount + j+kount_1p] = get_j(lx1, lx2, ly1, ly2, lz1, lz2)*
                            fc_gf(vib2v, vib1) * fc_gf(0, vib2); 
                sys_h[(j+kount_1p) * kount + i] = sys_h[i * kount + j +kount_1p]; 
            
            }
        }
    }    
    
//***********************************************//
//                        create 2p2p hamiltonian                                 //
//***********************************************/

    for (i = 0; i < kount_2p; i++) {  
        lx1 = nx_2p[i].lx;
        ly1 = nx_2p[i].ly;
        lz1 = nx_2p[i].lz;
        vib1 = nx_2p[i].vib; 
        lx1v = nx_2p[i].lxv;
        ly1v = nx_2p[i].lyv;
        lz1v = nx_2p[i].lzv;
        vib1v = nx_2p[i].vibv; 

        //diagonal part
        sys_h[(i+kount_1p) * kount + i + kount_1p] = (vib1 + vib1v) * hw; 

        for (j = 0; j < kount_2p; j++) {  
            if(i == j) continue; 

            lx2 = nx_2p[j].lx;
            ly2 = nx_2p[j].ly;
            lz2 = nx_2p[j].lz;
            vib2 = nx_2p[j].vib; 
            lx2v = nx_2p[j].lxv;
            ly2v = nx_2p[j].lyv;
            lz2v = nx_2p[j].lzv;
            vib2v = nx_2p[j].vibv; 
            
            // linker part 
            if(lx2v == lx1v && ly2v == ly1v && lz2v == lz1v && vib2v == vib1v){
                sys_h[(i+kount_1p) * kount + j+kount_1p] = get_j(lx1, lx2, ly1, ly2, lz1, lz2)*
                            fc_gf(0, vib1) * fc_gf(0, vib2); 
                sys_h[(j+kount_1p) * kount + i+kount_1p] = sys_h[(i+kount_1p) * kount + j +kount_1p]; 
            
            }

            // exchange part 
            if(lx2v == lx1 && ly2v == ly1 && lz2v == lz1 && 
                lx2 == lx1v && ly2 == ly1v && lz2 == lz1v){


                sys_h[(i+kount_1p) * kount + j+kount_1p] = get_j(lx1, lx2, ly1, ly2, lz1, lz2)*
                            fc_gf(vib2v, vib1) * fc_gf(vib1v, vib2); 
                sys_h[(j+kount_1p) * kount + i+kount_1p] = sys_h[(i+kount_1p) * kount + j +kount_1p]; 
            
            }
        }
    }    
    
    //print out the hamiltonian
    if (kount < 100) {
        FILE* stream = fopen((task_title + "_H.dat").c_str(),"w");
        int i,j; 
        
        //fprintf(stream, "Printing Matrix : \n");
        fprintf(stream, "\n %s\n", "Hamiltonian" );
        
        for( i = 0; i < kount; i++ ) {
            for( j = 0; j < kount; j++ ) fprintf(stream, " %6.2lf", sys_h[i*kount+j]);
            fprintf(stream, "\n" ); 
        }
    }
// start magma library

    // magma_queue_t       queue;

 
    // Initialize magma math library and queue


    ab_x = new double[spec_step];
    ab_y = new double[spec_step];
// cout << "hahahahahahah*** " << endl;
// parallel computing
// for(config = ista; config <= iend; config++){
    magma_diagonalization(kount, sys_h, 
        ux, uy, ab_x, ab_y, 
        spec_step, spec_start_ab, spec_end_ab, abs_lw); 
//}
/*     for(spec_point = 0; spec_point < spec_step; spec_point++ ){
        printf(" %lf \n", ab_x[spec_point]);
    }
 */
// cout << "hahahahahahah************************* " << endl;
// ***********************************************
    // sum over all absorption spectra and print
// ***********************************************
if(myid == MASTER){

    mab_x = new double[spec_step];
    mab_y = new double[spec_step];
}

    MPI_Reduce(ab_x, mab_x, spec_step, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(ab_y, mab_y, spec_step, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // MPI_REDUCE(ab_z, sum_ab_z, spec_step, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD);
    
    //this is wrong part

if(myid == MASTER){
    // cudaMemcpy( ab_x, d_ab_x, kount2* sizeof(double), cudaMemcpyDeviceToHost );
    for(spec_point = 0; spec_point < spec_step; spec_point++ ){
        ab_x[spec_point] = mab_x[spec_point]/conf_max;
        ab_y[spec_point] = mab_y[spec_point]/conf_max;
        // cout << ab_x[spec_point] << endl;
    }
    //print absorption spectrum
    FILE* stream = fopen((task_title + "_ab.dat").c_str(),"w");
    
        // fprintf(stream, "Printing Matrix : \n");
        fprintf(stream, "%s\n", "Energy A(g(w))" );
        fprintf(stream, "%s\n", "Energy System" );
        fprintf(stream, "%s\n\n", "cm +(-1) a.u." );
        
        for(spec_point = 0; spec_point < spec_step; spec_point++ ){
            energy = spec_start_ab + (spec_end_ab - spec_start_ab)/spec_step*(spec_point+1);
            // fprintf(stream, " %lf %lf %lf %lf %lf\n", energy, ab_x[spec_point]+ab_y[spec_point]+ab_z[spec_point],
                    // ab_x[spec_point], ab_y[spec_point], ab_z[spec_point]);
            fprintf(stream, " %lf %lf %lf %lf \n", energy, ab_x[spec_point]+ab_y[spec_point],
                    ab_x[spec_point], ab_y[spec_point]);
        }

    fclose(stream);

// cout << "hahahahahahah**************************************** " << endl;
    // ***********************************************
        // save the parameters that you use
    // ***********************************************
    ofstream file1(task_title + "_para.csv");
    file1 << "parameter, value" << endl;
    file1 << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;
    file1 << "task title, " << task_title << endl;
    file1 << "lattice_x, " << lattice_x << endl;
    file1 << "lattice_y, " << lattice_y << endl;
    file1 << "lattice_z, " << lattice_z << endl;
    file1 << "jo_x, " << jo_x << endl;
    file1 << "jo_y, " << jo_y << endl;
    file1 << "jo_z, " << jo_z << endl;
    file1 << "Huang-Rhys factor, " << s << endl;
    file1 << "vibrational energy (cm-1), " << hw << endl;
    file1 << "vibmax, " << vibmax << endl;
    file1 << "kount, " << kount << kount_1p << kount_2p << endl;
    file1 << "abs linewidth (cm-1), " << abs_lw  << endl;
    file1 << "dielectric, " << dielectric  << endl;
    file1 << "no_frenkel, " << no_frenkel  << endl;
    file1 << "periodic, " << periodic  << endl;    
    file1 << "two_state, " << two_state  << endl;
    file1 << "spec_step, " << spec_step  << endl;
    file1 << "spec_start_ab, " << spec_start_ab  << endl;
    file1 << "spec_end_ab, " << spec_end_ab  << endl;
    file1 << "spec_start_pl, " << spec_start_pl  << endl;
    file1 << "spec_end_pl, " << spec_end_pl  << endl;
    file1 << "conf_max, " << conf_max  << endl;
    file1 << "sigma, " << sigma  << endl;
    file1.close();
}

    // ***********************************************
        // free memory on the CPU and GPU and finalize magma library
    // ***********************************************
    


    delete[] sys_h;

    
    delete[] nx_1p; 
    delete[] nx_2p;


    delete[] ab_osc_x;
    delete[] ab_osc_y;
    // delete[] ab_osc_z;      
    delete[] ab_x;    
    delete[] ab_y;    
    // delete[] ab_z; 

    delete[] ux;    
    delete[] uy;    
    // delete[] uz;    

    delete[] pl_osc_x;
    delete[] pl_osc_y;
    // delete[] pl_osc_z;      
    delete[] pl_x;    
    delete[] pl_y;    
    // delete[] pl_z; 
    



    end = time(NULL);
    printf("\n>>> Done with the calculation in %f seconds.\n", difftime(end,start));
    
    return 0;
}
    
}
   
// Progress bar to keep updated on tcf
// void printProgress( int currentStep, int totalSteps )
// {
//     user_real_t percentage = (user_real_t) currentStep / (user_real_t) totalSteps;
//     int lpad = (int) (percentage*PWID);
//     int rpad = PWID - lpad;
//     fprintf(stderr, "\r [%.*s%*s]%3d%%", lpad, PSTR, rpad, "",(int) (percentage*100));
// }


